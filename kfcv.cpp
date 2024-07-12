#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "includes/eigen3/Eigen/Dense"
#include "includes/eigen3/Eigen/StdVector"

double misclassification_rate(std::vector<int> labels, std::vector<int> ground_truth_labels)
{

  /* Takes an array of labels and an array of ground truth labels and calculates the misclassification rate. */

  int incorrect = 0;

  std::vector<int>::iterator labels_it = labels.begin();
  std::vector<int>::iterator ground_truth_labels_it = ground_truth_labels.begin();

  for(; labels_it != labels.end() && ground_truth_labels_it != ground_truth_labels.end(); ++labels_it, ++ground_truth_labels_it)
  {
      if(*labels_it != *ground_truth_labels_it)
      {
        incorrect += 1;
      }
  }

  return (double) incorrect / labels.size();
}

std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > split(Eigen::MatrixXd dataset, int K)
{

  	/* Returns shuffled list of K Eigen::MatrixXd folds, split from input dataset. */

	int place = 0;

	//create temporary std::vector to hold rows of dataset
	std::vector<Eigen::Vector<double,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Vector<double,Eigen::Dynamic> > > temp;
	for(int i = 0; i < dataset.rows(); i++)
	{
		temp.push_back(dataset.row(i));
	}

	// shuffle std::vector

	auto random_number_generator = std::default_random_engine {};
	std::shuffle(std::begin(temp), std::end(temp), random_number_generator);

	// write shuffled rows into a new shuffled matrix

	Eigen::MatrixXd shuffled(dataset.rows(),dataset.cols());
	for( auto v : temp )
	{
		shuffled.row(place++) = v;
	}

	place = 0;

	std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > list; // does not like not regular ints as arguments, e.g. row len and fold len

	for(int i = 0; i < K; i++)
	{
		Eigen::MatrixXd fold(dataset.rows() / K,dataset.cols());

		for(int j = 0; j < dataset.rows() / K; j++)
		{
			Eigen::VectorXd x = shuffled.row(place++);
			fold.row(j) = x;
		}

		list.push_back(fold);
	}

	return list;
}

int run_counter = 1;

double kfcv(Eigen::MatrixXd dataset, int K, std::vector<int> (*classifier) (Eigen::MatrixXd validation, int validation_size, Eigen::MatrixXd training, int training_size, int length, bool verbose),bool verbose)
{
	/* Returns std::vector of error statistics from run of cross validation using given error function and classification function. */

	std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > folds = split(dataset,K);

	double total_error = 0;

	for(int i = 0; i < K; i++)
	{
		int length = dataset.rows() / K;
		int train_place = 0;
		int validation_place = 0;

		Eigen::MatrixXd validation(length * 1,dataset.cols());
		Eigen::MatrixXd train(length * (K-1),dataset.cols());

		int idx = 0;

		for(auto v : folds)
		{
			if(idx != i)
			{
				for(int j = 0; j < length; j++)
				{
					train.row(train_place++) = v.row(j);
				}
			}

			if(idx == i)
			{
				for(int j = 0; j < length; j++)
				{
					validation.row(validation_place++) = v.row(j);
				}
			}

			idx = idx + 1;
		}

		std::vector<int> truth_labels;

		idx = 0;

		for(int i = 0; i < validation.rows(); i++)
		{
			truth_labels.push_back(validation.coeff(i,0));
		}
		
		//std::cout << "validation rows: " << validation.rows() << "validation cols: " << validation.cols() << "\n" << validation << "\n";
		//std::cout << "trin rows: " << train.rows() << "train cols: " << train.cols() << "\n" << train << "\n";



		std::vector<int> predictions = classifier(validation, validation.rows(), train, train.rows(), train.cols(), false);

		double error = misclassification_rate(predictions,truth_labels);
		total_error += error;

		if(verbose)
		{
			if(i!=0)
			{
				printf("%d fold cross validation, fold %d error -> %f\n",K,run_counter++,(double) total_error/i);
			}else
			{
				printf("%d fold cross validation, fold %d error -> %f\n",K,run_counter++,(double) total_error);
			}
		}

	}

	//print("validation")

	return (double) total_error / (double) K;
}

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include "includes/eigen3/Eigen/Dense"
#include "includes/utils.h"
#include "includes/naive_bayes.h"
#include "includes/kfcv.h"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */


template<typename T> T load_csv(const std::string & sys_path)
{

  /* Returns csv file input as an Eigen matrix or vector. */

  std::ifstream in;
  in.open(sys_path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(in, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      while (std::getline(lineStream, cell, ',')) {
          values.push_back(std::stod(cell));
      }
      rows = rows + 1;
  }

  return Eigen::Map<const Eigen::Matrix<typename T::Scalar, T::RowsAtCompileTime, T::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);

  /* based on code from https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix/39146048 */
}

void driver(std::string sys_path_test, std::string sys_path_train, bool verbose, bool gaussian, bool categorical)
{

  /* Driver for a naive bayes classifier example. */

  Eigen::MatrixXd test = load_csv<Eigen::MatrixXd>(sys_path_test);

  if(verbose == true)
  {
      std::cout << "Test Data: " << sys_path_test << "\n";
      std::cout << test << "\n\n";
  }

  Eigen::MatrixXd train = load_csv<Eigen::MatrixXd>(sys_path_train);

  if(verbose == true)
  {
      std::cout << "Train Data: " << sys_path_train << "\n";
      std::cout << train << "\n\n";
  }

  if(gaussian == true)
  {
  	std::vector<int> predictions = gaussian_naive_bayes_classifier(test, test.rows(), train, train.rows(), train.cols(),verbose);
  	
    	int count = 0;
    	for(auto v : predictions)
    	{
        	if(verbose == true)
		{
			std::cout << "Row " << count << ": Class = " << v << "\n";
		} else
		{
			std::cout << v << "\n";
		}
		count++;
    	}
    	std::cout << "\n";
  	

  	if(verbose)
  	{
  		int num_folds = 10;
  		double result = kfcv(test,num_folds,&gaussian_naive_bayes_classifier,verbose);
		printf("\nmodel performance on new data: %f\n",result);
  	}

  } else if(categorical == true)
  {
	std::vector<int> predictions = categorical_naive_bayes_classifier(test, test.rows(), train, train.rows(), train.cols(), verbose);
	int count = 0;
    	for(auto v : predictions)
    	{
        	std::cout << "Row " << count << ": Class = " << v << "\n"; // add back after debugging
        	count++;
    	}
    	std::cout << "\n";

  	if(verbose)
  	{
  		int num_folds = 10;
  		double result = kfcv(test,num_folds,&categorical_naive_bayes_classifier,verbose);
		printf("\nmodel performance on new data: %f\n",result);
  	}
  }
}

int main(int argc, char ** argv)
{
	
  bool verbose = false;
  bool gaussian = false;
  bool categorical = false;

  if(argc == 1)
  {
    std::cout << "No arguments supplied.\n";
    std::cout << "Usage: ./naive-bayes-cli [train] [test] [options ..]\n";
    std::cout << "More info with: \"./naive-bayes-cli -h\"\n";
    return 1;
  }

  int counter = 1;

  while(counter < argc)
  {
    if(argv[counter][0] == '-' && argv[counter][1] == 'h') //&& argv[counter][2] == '\0'
    {
      std::cout << "Naive Bayes Cli (2021 Dec 9, compiled " << __TIMESTAMP__ << " " << __TIME__ << ")\n\n";
      std::cout << "usage: ./naive-bayes-cli [train] [test] [options ..]    read in train csv and test csv files from filesystem\n";
      std::cout << "   or: ./naive-bayes-cli -h                             displays help menu\n\n";
      std::cout << "Arguments:\n";
      std::cout << "   -h     Displays help menu\n";
      std::cout << "   -v     Displays output in verbose mode\n";
      std::cout << "   -g     Gaussian Naive Bayes\n";
      std::cout << "   -c     Categorical Naive Bayes\n";
      return 0;
    } else if(counter == 1 && !(valid_filepath(argv[1])))
    {
      std::cout << "Invalid filepath: " << argv[1] << "\n";
      std::cout << "Usage: ./naive-bayes-cli [train] [test] [options ..]\n";
      return 1;
    } else if(counter == 2 && !(valid_filepath(argv[1])))
    {
      std::cout << "Invalid filepath: " << argv[1] << "\n";
      std::cout << "Usage: ./naive-bayes-cli [train] [test] [options ..]\n";
      return 1;
    } else if(counter == 1 && (valid_filepath(argv[1])))
    {

    } else if(counter == 2 && (valid_filepath(argv[2])))
    {

    } else if(counter >= 3)
    {

      if(argv[counter][0] == '-' && argv[counter][1] == 'v' && argv[counter][2] == '\0')
      {
        verbose = true;
      } else if(argv[counter][0] == '-' && argv[counter][1] == 'g' && argv[counter][2] == '\0')
      {
	gaussian = true;
      } else if(argv[counter][0] == '-' && argv[counter][1] == 'c' && argv[counter][2] == '\0')
      {
      	categorical = true;
      } else
      {
        std::cout << "Unknown option argument: " << argv[counter] << "\n";
        std::cout << "More info with: \"./naive-bayes-cli -h\"\n";
        return 1;
      }
    } else
    {
      std::cout << "Unknown option argument: " << argv[counter] << "\n";
      std::cout << "More info with: \"./naive-bayes-cli -h\"\n";
      return 1;
    }



    counter = counter + 1;
  }

  if(gaussian || categorical)
  {
      driver(argv[2],argv[1],verbose,gaussian,categorical);
  } else
  {
  	printf("No classifier specificed. Please run with -g for gaussian or -c for categorical.\n");
  }

  return 0;
}

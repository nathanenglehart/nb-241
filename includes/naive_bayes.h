#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include "eigen3/Eigen/Dense"
#include "utils.h"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

int len(const Eigen::VectorXd&);
double mean(const Eigen::VectorXd&);
double standard_deviation(const Eigen::VectorXd&);
double gaussian_pdf(double, double, double);
std::vector<int> class_indicies(Eigen::MatrixXd, int);
std::vector<std::vector<double>> summarize_dataset(Eigen::MatrixXd, int);
std::vector<Eigen::MatrixXd> matricies_by_classification(Eigen::MatrixXd, int, int);
std::map<int, std::vector<std::vector<double>>> summarize_by_classification(Eigen::MatrixXd, int, int);
std::map<int, double> calculate_classification_probabilities(std::map<int, std::vector<std::vector<double>>>, Eigen::VectorXd, int, bool);
int predict(std::map<int, std::vector<std::vector<double>>>, Eigen::VectorXd, int, bool);
std::vector<int> gaussian_naive_bayes_classifier(Eigen::MatrixXd, int, Eigen::MatrixXd, int, int, bool);
std::vector<int> categorical_naive_bayes_classifier(Eigen::MatrixXd, int, Eigen::MatrixXd, int, int, bool);

#endif

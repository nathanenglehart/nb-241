#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include "eigen3/Eigen/Dense"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

float round_num(float);
bool valid_filepath(const std::string &);
double double_vector_list_lookup(std::vector<std::vector<double>>, int, int);
double get_eigen_index(Eigen::VectorXd, int);
bool compare_classification(const Eigen::VectorXd&, const Eigen::VectorXd&);
Eigen::MatrixXd sorted_rows_by_classification(Eigen::MatrixXd);

#endif

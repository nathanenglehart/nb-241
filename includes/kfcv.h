#ifndef KFCV_H
#define KFCV_H

double misclassification_rate(std::vector<int> labels, std::vector<int> ground_truth_labels);
std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > split(Eigen::MatrixXd dataset, int K);

double kfcv(Eigen::MatrixXd dataset, int K, std::vector<int> (*classifier) (Eigen::MatrixXd validation, int validation_size, Eigen::MatrixXd training, int training_size, int length, bool verbose),bool);

#endif

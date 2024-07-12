#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include "includes/eigen3/Eigen/Dense"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

float round_num(float number)
{

  /* Rounds number to 4 decimal points. */

  float ret = (int) (number * 10000 + .5);
  return (float) ret / 10000;
}


bool valid_filepath(const std::string & sys_path)
{

  /* Determines whether a system filepath is valid. */

  std::ifstream test(sys_path);
  if(!test)
  {
    return false;
  }

  return true;
}

double get_eigen_index(Eigen::VectorXd vector, int index)
{

 /* Returns the double value of a vector at the given index. */

 int place = 0;
 for(auto v : vector)
 {
   if(place == index)
   {
     return v;
   }
   place++;
 }

 exit(1);
}


double double_vector_list_lookup(std::vector<std::vector<double>> list, int first_index, int second_index)
{

 /* Finds the double value located in a double vector list located at first index, second index. */

 int first_count = 0;
 for(auto v : list)
 {
   if(first_index == first_count)
   {
     int second_count = 0;
     for(auto w : v)
     {
       if(second_count == second_index)
       {
         return w;
       }
       second_count++;
     }
   }
   first_count++;
 }

 exit(1);
}

bool compare_classification(const Eigen::VectorXd& l, const Eigen::VectorXd& r)
{

  /* Compares the first index of one vector to the first index of another vector. */

  return l(0) < r(0);
}

Eigen::MatrixXd sorted_rows_by_classification(Eigen::MatrixXd X)
{

  /* Sorts the input matrix according to the first column entry. */

  std::vector<Eigen::VectorXd> vec;

  for(int64_t i = 0; i < X.rows(); ++i)
  {
    vec.push_back(X.row(i));
  }

  std::sort(vec.begin(), vec.end(), &compare_classification);

  for(int64_t i = 0; i < X.rows(); ++i)
  {
    X.row(i) = vec[i];
  }

  return X;
}

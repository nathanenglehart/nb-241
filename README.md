# Naive Bayesian Classifier Algorithm
## Authors
Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021)

## Usage
To clone and run this classifier so that it can be run on a dataset, please run the following. 

```bash
git clone https://github.com/nathanenglehart/naive-bayes-cpp-241
cd naive-bayes-cpp-241
make
```

The program is meant to be run as the below, where train and test are the paths to the train and test csv files.

```bash
./naive-bayes-cli [train] [test] [options...]
```

For a help menu, please run:

```bash
./naive-bayes-cli -h
```

which displays:

```
Naive Bayes Cli (2021 Dec 9, compiled Wed Jul 20 15:05:41 2022 15:06:06)

usage: ./naive-bayes-cli [train] [test] [options ..]    read in train csv and test csv files from filesystem
   or: ./naive-bayes-cli -h                             displays help menu

Arguments:
   -h     Displays help menu
   -v     Displays output in verbose mode
   -g     Gaussian Naive Bayes
   -c     Categorical Naive Bayes
```

To run this program in verbose mode, please run:

```bash
./naive-bayes-cli [train] [test] -v 
```

## Install
To install this program to your posix standard system, please run the following.

```bash
git clone https://github.com/nathanenglehart/naive-bayes-cpp-241
cd naive-bayes-cpp-241
make
sudo cp naive-bayes-cli /usr/local/bin/naive-bayes-cli
sudo chmod 0755 /usr/local/bin/naive-bayes-cli
```

The program can then be run from any location on your system, as in the below.

```bash
naive-bayes-cli [train] [test] [options...]
```

## Uninstall
To uninstall this program from your system, run the following.

```bash
sudo rm /usr/local/bin/naive-bayes-cli
```

## Notes

We are currently including the eigen3 linear algebra library folder within this program for a simpler installation process. See here: [https://eigen.tuxfamily.org/index.php?title=Main_Page](https://eigen.tuxfamily.org/index.php?title=Main_Page).

## References

Barber, David. (2016). Bayesian Reasoning and Machine Learning. Cambridge University Press.

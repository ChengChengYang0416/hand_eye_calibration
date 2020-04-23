#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>

#define MAXBUFSIZE  ((int) 1e6)

using namespace std;
using namespace Eigen;

// declare for row and column of matrix
int cols = 0, rows = 0;

MatrixXd readMatrix(const char *filename){

    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);

    while (! infile.eof()){
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof()){
          stream >> buff[cols*rows+temp_cols++];
        }

        if (temp_cols == 0){
          continue;
        }

        if (cols == 0){
          cols = temp_cols;
        }
        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        result(i,j) = buff[ cols*i+j ];
      }
    }

    return result;
}

int main(int argc, char** argv){

  // get position in both robotic frame and camera frame from calibration.txt
  MatrixXd m;
  const char *file = "/home/ncrl/qt_ws/src/Sensing_int_sys/hand_eye_calibration/src/calibration.txt";
  m = readMatrix(file);

  // depart position to robotic frame and camera frame
  MatrixXd m1(rows, 3);
  MatrixXd m2(rows, 3);
  for (int i = 0; i < rows; i++){
    m1.block<1, 3>(i, 0) = m.block<1, 3>(i, 0);
    m2.block<1, 3>(i, 0) = m.block<1, 3>(i, 3);
  }
  MatrixXd robotics_pos = m1.transpose();
  MatrixXd camera_pos = m2.transpose();

  // calculate p, p', qi, qi' from eq.6, eq.4, eq.7, eq.8
  Vector3d p(0, 0, 0);
  Vector3d p_prime(0, 0, 0);
  for (int i = 0; i < rows; i++){
    p += camera_pos.block<3, 1>(0, i);
    p_prime += robotics_pos.block<3, 1>(0, i);
  }
  p = p/rows;
  p_prime = p_prime/rows;

  MatrixXd q(3, rows);
  MatrixXd q_prime(3, rows);
  for (int i = 0; i < rows; i++){
    q.block<3, 1>(0, i) = camera_pos.block<3, 1>(0, i) - p;
    q_prime.block<3, 1>(0, i) = robotics_pos.block<3, 1>(0, i) - p_prime;
  }

  // calculate H, R_hat, T_hat from eq.11, eq.13, eq,10
  MatrixXd H(3, 3);
  H << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  for (int i = 0; i < rows; i++){
    H += q*(q_prime.transpose());
  }

  JacobiSVD<MatrixXd> svd(H, ComputeThinU | ComputeThinV);
  Matrix3d X;
  Matrix3d R_hat;
  Vector3d T_hat;
  MatrixXd T(4, 4);
  Vector4d fourth(0, 0, 0, 1);
  X << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  X = svd.matrixV()*(svd.matrixU().transpose());
  Matrix3d inverse;
  bool invertible;
  double determinant;
  X.computeInverseAndDetWithCheck(inverse, determinant, invertible);
  if (determinant != -1){
    R_hat = X;
    T_hat = p_prime - R_hat*p;
    T.block<3, 3>(0, 0) = R_hat;
    T.block<3, 1>(0, 3) = T_hat;
    T.block<1, 4>(3, 0) = fourth.transpose();
    cout << T << endl;
  }
  else {
    cout << "The algorithm fails." << endl;
  }

  return 0;
}

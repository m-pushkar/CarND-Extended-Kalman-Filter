#include "kalman_filter.h"
#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) 
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() 
{
  // Predict state //
  x_ = F_*x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) 
{
  // Update using Kalman filter equations for laser //
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  
  // New estimates //
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;   
}

void KalmanFilter::UpdateEKF(const VectorXd &z) 
{
  // Update using EKF for radar //
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  // Pre-compute repeated calculations //
  float rho = sqrt (px*px + py*py);
  float phi = atan2(py, px); 
  
  //if(fabs(rho) < 0.00001)
  //{
   // float rho_dot = 0.00001; // rho_dot = 0 but set to 0.00001 to avoid division by 0 //
 // }
 // else
 // {
  float rho_dot = (px * vx + py * vy)/rho;
  //}
  
  VectorXd hx(3);
  hx << rho, phi, rho_dot;
  
  VectorXd y = z - hx;
  float angle = y(1);
  const float PI = 3.14159265;
  if (angle > PI || angle < -PI)
  {
    float new_angle = std::fmod(angle, 2.0*PI);
    angle = new_angle < 0 ? new_angle + 2.0 * PI : new_angle;
    y(1) = angle;
    cout << " Normalizing angle " << endl;
  }
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * H_.transpose() * Si;
  
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;     
}
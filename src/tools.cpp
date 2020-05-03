#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // Calculate RMSE //
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
    cout << "Invalid Data" << endl; 
    return rmse;
  }
  
  // Squared residual //
  for (unsigned int i = 0; i < estimations.size(); ++i)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }
  
  // Mean // 
  rmse = rmse/estimations.size();
  
  // RMSE //
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
  MatrixXd Hj(3,4);
  
  // State parameters //
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  // Pre-compute repeated parameters //
  float a1 = px*px + py*py;
  float a2 = sqrt(a1);
  float a3 = a1*a2;
  
  // Check division by zero condition // 
  if (fabs(a1) < 0.001)
  {
    cout << " Error - division by zero " << endl;
    return Hj;
  }
  
  // Jacobian matrix //
  Hj << (px/a2), (py/a2), 0, 0,
       -(py/a1), (px/a1), 0, 0,
        py*(vx*py - vy*px)/a3, px*(px*vy - py*vx)/a3, px/a2, py/a2;
  
  return Hj; 
}

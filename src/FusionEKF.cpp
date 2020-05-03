#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/** 
* Constructor 
*/
FusionEKF::FusionEKF() 
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Initializing matrices //
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // Measurement covariance matrix - laser //
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar //
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  // Measurement matrix - laser //
  H_laser_ << MatrixXd::Identity(2,4);
  Hj_ << MatrixXd::Zero(3,4);
  
  // Initial state covariance matrix //
  ekf_.P_ = MatrixXd (4,4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0 ,0, 1000;
  
  // Initial transition matrix //
  ekf_.F_ = MatrixXd (4,4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0 ,0, 1; 
  
  // Process covariance matrix //
  ekf_.Q_ = MatrixXd (4,4);
  ekf_.Q_ << MatrixXd::Zero(4,4);
  
  // State vector //
  ekf_.x_ = VectorXd (4);
  ekf_.x_ << VectorXd::Zero(4);
  
  // ekf_.Init(ekf_.x_, ekf_.Q_, ekf_.F_, ekf_.P_, H_laser, R_laser, R_radar)
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
  /**
   * Initialization
   */
  
  if (!is_initialized_) 
  {
    // First measurement //
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
      // Polar to cartesian conversion, radar initialization //
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_(0) = rho*cos(phi);
      ekf_.x_(1) = rho*sin(phi);
      ekf_.x_(2) = rho_dot*cos(phi);
      ekf_.x_(3) = rho_dot*sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
    {
      // Laser initialization // 
      ekf_.x_(0) = measurement_pack.raw_measurements_[0];
      ekf_.x_(1) = measurement_pack.raw_measurements_[1];
      ekf_.x_(2) = 0;
      ekf_.x_(3) = 0;

    }
    
    previous_timestamp_ = measurement_pack.timestamp_;

    // Done initializing, no need to predict or update //
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  
  // Compute time elapsed in seconds //
  float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  float dt2 = dt*dt;
  float dt3 = dt2*dt;
  float dt4 = dt3*dt;
  
  // Modify F matrix to integrate time //
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  
  // Measurement noise //
  float noise_ax = 9;
  float noise_ay = 9;
  
  // Process covariance matrix // 
  ekf_.Q_ <<   dt4/4*noise_ax, 0,              dt3/2*noise_ax, 0,
               0,              dt4/4*noise_ay, 0,              dt3/2*noise_ay,
               dt3/2*noise_ax, 0,              dt2*noise_ax,   0,
               0,              dt3/2*noise_ay, 0,              dt2*noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
  {    
    // Radar updates //
    ekf_.R_ = R_radar_;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else 
  {
    // Laser updates //
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // Print the output //
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

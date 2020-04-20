#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5*3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_aug_ = n_x_ + 2; 
    //This will hold the size for augmented state size
  lambda_ = 3 - n_aug_;

  
  sigma_points_ = MatrixXd( n_aug_, 2*n_aug_ + 1);
  Xsig_pred_ = MatrixXd( n_x_, 2*n_aug_ + 1);
    //Initializing the matrix for storing the sigma points, and the predicted sigma points

  //The weights for prediction the covariances and mean can be initialized here. They should not change

  weights_ = VectorXd( 2*n_aug_+1 );
  weights_(0) = lambda_/( lambda_ + n_aug_ );
  for( size_t i = 1; i < 2*n_aug_+1; ++i ) {
    weights_(i) = 0.5/( lambda_ + n_aug_ );    
  }

  //Below we will initialize the covariance matrices for both sensors. As the are constant, 
  //We will initialize then there

  R_radar_ = MatrixXd( 3, 3 );
  R_radar_.fill(0.0);
  
  R_radar_(0,0) = std_radr_*std_radr_;
  R_radar_(1,1) = std_radphi_*std_radphi_;
  R_radar_(2,2) = std_radrd_*std_radrd_;
  
  R_lidar_ = MatrixXd( 2,2 );
  R_lidar_.fill(0.0);
  R_lidar_( 0,0 ) = std_laspx_*std_laspx_;
  R_lidar_( 1,1 ) = std_laspy_*std_laspy_;
  
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if( is_initialized_ ) {


    bool laser = false, radar = false;
    if( ( meas_package.sensor_type_ == MeasurementPackage::LASER ) && use_laser_ ) {
      laser = true;
    }else if( ( meas_package.sensor_type_ == MeasurementPackage::RADAR ) &&  use_radar_ ) {
      radar = true;
    }else {
      return;
      //Do nothing and return, invalid sensor_type_
    }

    //If we reached this point, so a valid sensor type was found, we are able to predict them update
    double dt = (meas_package.timestamp_ - time_us_)/1e6;
    Prediction( dt );

    //Ready for update
    if( laser ) {
      UpdateLidar( meas_package );
    }else {
      UpdateRadar( meas_package );
    }

    time_us_ = meas_package.timestamp_;

    //std::cout << "x: " << x_ << std::endl;

  }else {

    //First, 
    if( ( meas_package.sensor_type_ == MeasurementPackage::LASER ) ) {
      //All parameters ready, let's mark as initialized;
      is_initialized_ = true;
      time_us_ = meas_package.timestamp_;

      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);

      x_(0) = px;
      x_(1) = py;
      x_(2) = 0.0;
      x_(3) = 0.0;
      x_(4) = 0.0;

      P_.setIdentity();
      P_ *= 1.0;

      //Melhor setup
      P_(0.0) = std_laspx_*std_laspx_*0.33;
      P_(1.1) = std_laspy_*std_laspy_*0.33;
      P_(2.2) = 2.0;
      P_(3.3) = 0.01;
      P_(4.4) = 0.7;

      // P_(0.0) = std_laspx_*std_laspx_*100;
      // P_(1.1) = std_laspy_*std_laspy_*100;
      // P_(2.2) = 1;
      // P_(3.3) = 0.5;
      // P_(4.4) = 1e-6;
    }
  }
}

void UKF::Prediction(double delta_t) {

  //First, we need to generate the correct number of sigma points
  GenerateSigmaPoints();

  //Updating the sigma points. As we wrote a function for updating the state, let's use it
  for( size_t i = 0; i < 2*n_aug_ + 1; ++i ) {
    const Eigen::VectorXd sig_temp( sigma_points_.col(i) );
      //Just an const ref for alias. It will make easier to read without affecting the performance
    
    Xsig_pred_.col(i) = UpdateState( sig_temp.head( n_x_ ), sig_temp.tail( n_aug_ - n_x_ ), delta_t );
      //the first 5 elements are the state, while the last two are the "nu", or the accelerations for the CTRV
  }

  //Now we will predict the mean and covariance

  // predict state mean
  x_.fill(0);
  for( size_t i = 0; i < 2 * n_aug_ + 1; ++i ) {
      x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0);    
  for( size_t i = 0; i < 2 * n_aug_ + 1; ++i ) {
    VectorXd x_diff = ( Xsig_pred_.col(i) - x_ );
    
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  //Now we reached the end of the state. x_ and P_ have the updated values using the unscented transformation

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
  const size_t n_z = meas_package.raw_measurements_.size();

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd( n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd( n_z );
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd( n_z, n_z );


  // transform sigma points into measurement space
  double px, py;
  
  for( size_t i = 0; i < 2*n_aug_+1; ++i ) {

    px      = Xsig_pred_.col(i)(0);
    py      = Xsig_pred_.col(i)(1);

    Zsig(0,i) = px;
    Zsig(1,i) = py;
  }
  
  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for( size_t i = 0; i < 2*n_aug_+1; ++i ) {
      z_pred = z_pred + weights_(i)*Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for( size_t i = 0; i < 2*n_aug_+1; ++i ) {
      VectorXd diff = ( Zsig.col(i) - z_pred );
      S = S + weights_(i)*diff*diff.transpose();
  }
  S = S+R_lidar_;

  //Now the firnal part of the update
  
  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd( n_x_, n_z );
  Tc.fill(0.0);
  for( size_t i = 0; i < 2 * n_aug_ + 1; ++i ) {

      // residual
      VectorXd z_diff = Zsig.col(i) - z_pred;
      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      // angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  
  // calculate Kalman gain K;
  MatrixXd K( n_x_, n_z );
  K = Tc*S.inverse();

  // update state mean and covariance matrix
  x_ = x_ + K * ( meas_package.raw_measurements_ - z_pred );
  P_ = P_ - K*S*K.transpose();
}


void UKF::UpdateRadar(MeasurementPackage meas_package) {
  const size_t n_z = meas_package.raw_measurements_.size();

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd( n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd( n_z );
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd( n_z, n_z );


  // transform sigma points into measurement space
  double px, py, v, yaw, yawRate, rad, bearing, radVel;
  
  for( size_t i = 0; i < 2*n_aug_+1; ++i ) {

    px      = Xsig_pred_.col(i)(0);
    py      = Xsig_pred_.col(i)(1);
    v       = Xsig_pred_.col(i)(2);
    yaw     = Xsig_pred_.col(i)(3);
    yawRate = Xsig_pred_.col(i)(4);
    
    rad = sqrt( pow( px,2 ) + pow( py,2 ) );
    bearing = atan2( py, px );
    radVel = ( px*cos( yaw )*v + py*sin( yaw )*v )/rad;
    
    Zsig(0,i) = rad;
    Zsig(1,i) = bearing;
    Zsig(2,i) = radVel;
  }
  
  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for( size_t i = 0; i < 2*n_aug_+1; ++i ) {
      z_pred = z_pred + weights_(i)*Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for( size_t i = 0; i < 2*n_aug_+1; ++i ) {
      VectorXd z_diff = ( Zsig.col(i) - z_pred );

      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      S = S + weights_(i)*z_diff*z_diff.transpose();
  }
  
  S = S+R_radar_;

  //Now the firnal part of the update


  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd( n_x_, n_z );
  Tc.fill(0.0);
  for( size_t i = 0; i < 2 * n_aug_ + 1; ++i ) {
            // residual
      VectorXd z_diff = Zsig.col(i) - z_pred;
      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      // angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;


  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

}

Eigen::VectorXd UKF::UpdateState( const Eigen::VectorXd &x, const Eigen::VectorXd & nu, double dt ) const {

  Eigen::VectorXd xOut(5);
  
  xOut.fill(0);
  if( fabs( x(4) ) > 0.001 ) {
      xOut(0) = x(0) + (x(2)/x(4))*(  sin( x(3)+x(4)*dt) - sin(x(3) ) );
      xOut(1) = x(1) + (x(2)/x(4))*( -cos( x(3)+x(4)*dt) + cos(x(3) ) );      
  }else {
      xOut(0) = x(0) + x(2)*cos( x(3) )*dt;
      xOut(1) = x(1) + x(2)*sin( x(3) )*dt;
  }
  
  xOut(2) = x(2);
  xOut(3) = x(3)+ x(4)*dt;
  xOut(4) = x(4);
  
  VectorXd nuPart(5);
  
  nuPart(0) = 0.5*(dt*dt)*cos( x(3) )*nu(0);
  nuPart(1) = 0.5*(dt*dt)*sin( x(3) )*nu(0);
  nuPart(2) = dt*nu(0);
  nuPart(3) = 0.5*(dt*dt)*nu(1);
  nuPart(4) = dt*nu(1);
  
  xOut += nuPart;
  
  return xOut;
}

void UKF::GenerateSigmaPoints( ) {
  //Gerating the sigma points
  //Creating the augmented state from the current filter state

  Eigen::VectorXd x_aug( n_aug_ );
  x_aug.fill(0.0);          
    //Filling the x_aug with zeros as the augmented state additional state represents the mean for the noises
  x_aug.head( n_x_ ) = x_;

  //Creating the augmented covariance matrix
  MatrixXd P_aug = MatrixXd( n_aug_,  n_aug_ );

  P_aug.fill(0.0);
  P_aug.block< 5, 5 >( 0, 0 ) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //Now effectivelly calculating the sigma points
  
  sigma_points_.col(0) = x_aug;
    //The first col. is the current mean, extracted by the filter state

    // set remaining sigma points
  for ( size_t i = 0; i < n_aug_; ++i) {
    sigma_points_.col(i+1)        = x_aug + sqrt( lambda_+ n_aug_ ) * A.col(i);
    sigma_points_.col(i+1+n_aug_) = x_aug - sqrt( lambda_+ n_aug_ ) * A.col(i);
  }
}
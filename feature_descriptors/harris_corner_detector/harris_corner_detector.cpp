
/* Tutorial

http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

*/


#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo( int, void* );

/** @function main */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  src = imread("./1.jpg", -1 );
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  /// Create a window and a trackbar
  namedWindow( source_window, WINDOW_AUTOSIZE );
  createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );
  imshow( source_window, src );

  cornerHarris_demo( 0, 0 );

  waitKey(0);
  return(0);
}

/** @function cornerHarris_demo */
void cornerHarris_demo( int, void* )
{

  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src.size(), CV_32FC1 );

  /// Detector parameters
  int blockSize = 2; //blockSize - It is the size of neighbourhood considered for corner detection
  int apertureSize = 3; // apertureSize - Aperture parameter of Sobel derivative used.
  double k = 0.04; //Harris detector free parameter in the equation

  /// Detecting corners
  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

  namedWindow( "cornerHarris dst", WINDOW_AUTOSIZE );
  imshow( "cornerHarris dst", dst );
  imwrite("./cornerHarris_dst.jpg", dst);

  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );

  namedWindow( "normalize", WINDOW_AUTOSIZE );
  imshow( "normalize", dst_norm );
  imwrite("./normalize.jpg", dst_norm);

  convertScaleAbs( dst_norm, dst_norm_scaled );

  namedWindow( "Abs", WINDOW_AUTOSIZE );
  imshow( "Abs", dst_norm_scaled );
  imwrite("./abs.jpg", dst_norm_scaled);

  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
               circle( src, Point( i, j ), 10,  Scalar(0,0,255), 2, 8, 0 );
              }
          }
     }
  /// Showing the result
  namedWindow( corners_window, WINDOW_AUTOSIZE );
  imshow( corners_window, src );
  imwrite("./harris_corner_detector.jpg", src);

#if 0
	/* refine the corners subpixel level*/
  /// Set the neeed parameters to find the refined corners
vector<Point2f> corners;
  Size winSize = Size( 5, 5 );
  Size zeroZone = Size( -1, -1 );
  TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

  /// Calculate the refined corner locations
  cornerSubPix( src_gray, corners, winSize, zeroZone, criteria );

  for( int k = 0; k < corners.size(); k++ )
  { 
	circle( src, corners[k], 10,  Scalar(0,255,0), 2, 8, 0 );
  }

  namedWindow( "subpixel corners_window", WINDOW_AUTOSIZE );
  imshow( "subpixel corners_window", src );
  imwrite("./subpixel.jpg", src);
#endif
}

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

RNG rng(12345);

Mat GoofFeatures2Track(Mat src, Mat src_gray)
{

	int maxCorners = 100;
	int maxTrackbar = 100;

	vector<Point2f> corners;
	  double qualityLevel = 0.01;
	  double minDistance = 10;
	  int blockSize = 3;
	  bool useHarrisDetector = false;
	double k = 0.04;

  /// Copy the source image
  Mat copy;
  copy = src.clone();

  /// Apply corner detection
  goodFeaturesToTrack( src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k );


  /// Draw corners detected
  cout<<"** Number of corners detected: "<<corners.size()<<endl;
  int r = 4;
  for( int i = 0; i < corners.size(); i++ )
  { 
	circle( copy, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 ); 
	cout<<" ***corner["<<i <<"]: "<< corners[i]<<endl;
  }

  return copy;
}

int main(int argc, char* argv[])
{
	/* The input image can be any 8-bit or 32-bit (i.e., 8U or 32F), single-channel image.  */
	Mat src = imread("./1.jpg");
  	namedWindow( "src", cv::WINDOW_AUTOSIZE );
  	imshow( "src", src );
	cout << "src type: " <<src.type()<< endl;
/*
	Mat src2 = imread("./2.jpg");
  	namedWindow( "src2", cv::WINDOW_AUTOSIZE );
  	imshow( "src2", src2 );
	cout << "src2 type: " <<src2.type()<< endl;
*/

	Mat src_gray;
    cvtColor( src, src_gray, CV_BGR2GRAY );
  	namedWindow( "src_gray", cv::WINDOW_AUTOSIZE );
  	imshow( "src_gray", src_gray );
	cout << "src_gray type: " <<src_gray.type()<< endl;
/*
	Mat src2_gray;
    cvtColor( src2, src2_gray, CV_BGR2GRAY );
  	namedWindow( "src2_gray", cv::WINDOW_AUTOSIZE );
  	imshow( "src2_gray", src2_gray );
	cout << "src2_gray type: " <<src2_gray.type()<< endl;
*/
	Mat result1, result2;
	result1 = GoofFeatures2Track(src, src_gray);
    namedWindow( "result1", CV_WINDOW_AUTOSIZE );
    imshow( "result1", result1 );
	imwrite("./good_features_to_track.jpg", result1);
/*
	result2 = GoofFeatures2Track(src2, src2_gray);
    namedWindow( "resul2", CV_WINDOW_AUTOSIZE );
    imshow( "result2", result1 );
*/
	cvWaitKey(0);
}


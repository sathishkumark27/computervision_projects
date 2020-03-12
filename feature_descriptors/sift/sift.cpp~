#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;



/** @function main */
int main(  )
{	

  Mat img_1 = imread( "./1.jpg", IMREAD_GRAYSCALE );
  Mat img_2 = imread( "./2.jpg", IMREAD_GRAYSCALE );

 	//Size size(512, 512);

  //resize(img_1, img_1, size);
  //resize(img_2, img_2, size);

  	//namedWindow( "11", cv::WINDOW_AUTOSIZE );
  	//imshow( "1", img_1 );

  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  Ptr<SIFT> detector = SIFT::create();

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );

  //-- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
  imshow("Keypoints 2", img_keypoints_2 );
  imwrite("./Keypoints_1.jpg", img_keypoints_1);
  imwrite("./Keypoints_2.jpg", img_keypoints_2);

	Mat descriptors_1, descriptors_2;

	detector->compute(img_1, keypoints_1, descriptors_1);
	detector->compute(img_2, keypoints_2, descriptors_2);

  //-- Show descriptors
  imshow("descriptors 1", descriptors_1 );
  imshow("descriptors 2", descriptors_2 );
  imwrite("./descriptors_1.jpg", descriptors_1);
  imwrite("./descriptors_2.jpg", descriptors_2);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	Mat result;
  drawMatches( img_1, keypoints_1, img_1, keypoints_2, matches, result );
  imshow("matching", result );
  imwrite("./matchng.jpg", result);

 

  waitKey(0);

  return 0;
  }

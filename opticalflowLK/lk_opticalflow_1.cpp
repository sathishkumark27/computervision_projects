#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

std::vector<cv::Point2f> corners;
std::vector<cv::Point2f> corners_b;
double qualityLevel = 0.01;
double minDistance = 10;
int blockSize = 3;
bool useHarrisDetector = true;
double k = 0.04;
int maxCorners = 200;
int maxTrackbar = 100;

void MotionDetection(cv::Mat frame1, cv::Mat frame2)
{
    cv::Mat prev, next;
    cvtColor(frame1, prev, CV_BGR2GRAY); 
    cvtColor(frame2, next, CV_BGR2GRAY); 
    //cout<<"OK4"<<endl;
    goodFeaturesToTrack( prev, 
            corners,
            maxCorners,
            qualityLevel,
            minDistance,
            cv::Mat(),
            blockSize,
            useHarrisDetector,
            k );
            //cout<<"OK5"<<endl;
    cornerSubPix(prev, 
            corners,
            cvSize( 10, 10 ) ,
            cvSize( -1, -1 ), 
            cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );
            //cout<<"OK6"<<endl;
    std::vector<uchar> features_found;
    features_found.reserve(maxCorners);
    std::vector<float> feature_errors;
    feature_errors.reserve(maxCorners);
    //cout<<"OK7"<<endl;
    calcOpticalFlowPyrLK(prev, next, corners, corners_b, features_found, 
            feature_errors, cvSize( 10, 10 ), 5, cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0);
    //cout<<"OK8"<<endl;
    //IplImage *g = next;

    for( int i = 0; i < maxCorners; ++i )
    {
        /*CvPoint p0 = cvPoint( cvRound( corners[i].x ), cvRound( corners[i].y ) );
        CvPoint p1 = cvPoint( cvRound( corners_b[i].x ), cvRound( corners_b[i].y ) );
        //cout<<"OK9"<<endl;
        line( next, p0, p1, CV_RGB(255,0,0), 3, CV_AA );*/
        //cout<<"OK10"<<endl;

        line(frame2, Point(cvRound(corners[i].x), cvRound(corners[i].y)), 
             Point(cvRound( corners_b[i].x), cvRound( corners_b[i].y)),
				CV_RGB(0, 255, 0));
        cout<<"p0"<< Point(cvRound(corners[i].x), cvRound(corners[i].y))<<endl;
        cout<<"p1"<< Point(cvRound( corners_b[i].x), cvRound( corners_b[i].y))<<endl;
    }
    //cv::Mat rs(g);
    imshow( "result window", frame2 ); 
    //cout<<"OK11"<<endl; 
    int key = cv::waitKey(5);
}

int main(int argc, char* argv[])
{
    cv::VideoCapture cap(1); 
    if(!cap.isOpened())              
    {
        std::cout<<"[!] Error: cant open camera!"<<std::endl;
        return -1;
    }
    cv::Mat edges;
    cv::namedWindow("result window", 1);
    (void)system("v4l2-ctl -d /dev/video1 -c exposure_auto=1"); //logitech
    cv::Mat frame, frame2;
    (void)system("v4l2-ctl -d /dev/video1 -c exposure_absolute=10");
    cap >> frame;
    
    for (int i=0; ; i++)
    {
        (void)system("v4l2-ctl -d /dev/video1 -c exposure_absolute=10");
        //<<"OK1"<<endl;
        cap >> frame2;
        MotionDetection(frame, frame2);
        //cout<<"OK2"<<endl;
        frame2.copyTo(frame);
        //cout<<"OK3"<<endl;
    }
    return 0;
}
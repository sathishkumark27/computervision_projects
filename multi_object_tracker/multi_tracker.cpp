#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
#include "Hungarian.h"
#include "KalmanTracker.h"
#include <iostream>
#include <stdlib.h>
using namespace std;
using namespace cv;

#if 0
vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

Ptr<Tracker> createTrackerByName(string trackerType) 
{
  Ptr<Tracker> tracker;
  if (trackerType ==  trackerTypes[0])
    tracker = TrackerBoosting::create();
  else if (trackerType == trackerTypes[1])
    tracker = TrackerMIL::create();
  else if (trackerType == trackerTypes[2])
    tracker = TrackerKCF::create();
  else if (trackerType == trackerTypes[3])
    tracker = TrackerTLD::create();
  else if (trackerType == trackerTypes[4])
    tracker = TrackerMedianFlow::create();
  else if (trackerType == trackerTypes[5])
    tracker = TrackerGOTURN::create();
  else if (trackerType == trackerTypes[6])
    tracker = TrackerMOSSE::create();
  else if (trackerType == trackerTypes[7])    
    tracker = TrackerCSRT::create();
  else {
    cout << "Incorrect tracker name" << endl;
    cout << "Available trackers are: " << endl;
    for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
      std::cout << " " << *it << endl;
  }
  return tracker;
}
#endif

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;

int display = 1;
int total_frames = 0;
double total_time = 0.0;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
  cout<<"bb_test :"<<bb_test<<endl;
  cout<<"bb_gt :"<<bb_gt<<endl;
	float in = (bb_test & bb_gt).area();
  cout<<"in :"<<in<<endl;
	float un = bb_test.area() + bb_gt.area() - in;
  cout<<"un :"<<un<<endl;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

vector<cv::Rect> getBoundingBoxs(cv::Mat image, cv::Rect ROI,int count)
{

  cv::Mat grayimage;
  cv::Mat bw_image;
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat gaussian_image; 
  cv::Size ksize; //Gaussian kernel size. If equal to zero, then the kerenl size is computed from the sigma
  ksize.width = 0;
  ksize.height = 0;  
  std::vector<cv::Point2f> distorted_points;
  vector<double> areas;
  vector<cv::Rect> blobrect;
  vector<char> blobcolor; // R=0, G=0, everything else -1

  //defaults values
  double min_blob_area = 20;
  double gaussian_sigma = 0.6;
  double max_blob_area = 400;
  double max_width_height_distortion = 0.5;
  double max_circular_distortion = 0.5;
  

  cvtColor(image, grayimage, CV_BGR2GRAY);

  cv::threshold(grayimage(ROI), bw_image, 159, 255, cv::THRESH_TOZERO);

  // Gaussian blur the image  
  GaussianBlur(bw_image.clone(), gaussian_image, ksize, gaussian_sigma, gaussian_sigma, cv::BORDER_DEFAULT);
  
  cv::findContours(gaussian_image.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  for (unsigned i = 0; i < contours.size(); i++)
  {
      double area = cv::contourArea(contours[i]); // Blob area
      cv::Rect rect = cv::boundingRect(contours[i]); // Bounding box
      double radius = (rect.width + rect.height) / 4; // Average radius

      char ledx[10] = {0};
        //char ledx1[10] = {0};

      

      cv::Moments mu;
      mu = cv::moments(contours[i], false); // gives 3 degress of moments m00 to m22
      cv::Point2f mc;
      mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00) + cv::Point2f(ROI.x, ROI.y);  //center of then blob w.r.t the ROI
      // Look for round shaped blobs of the correct size
      snprintf (ledx, 10, "%u", (unsigned)mc.x);     
      cv::putText(image, ledx, mc, FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
      

    
      if ((area >= min_blob_area) && (area <= max_blob_area)
          && (std::abs(1 - std::min((double)rect.width / (double)rect.height, (double)rect.height / (double)rect.width))
              <= max_width_height_distortion)
          && (std::abs(1 - (area / (CV_PI * std::pow(rect.width / 2, 2)))) <= max_circular_distortion)
          && (std::abs(1 - (area / (CV_PI * std::pow(rect.height / 2, 2)))) <= max_circular_distortion))
      {
            distorted_points.push_back(mc);
            areas.push_back(area);
            blobrect.push_back(rect);
      }
  }

  char imname[100] = {0};
  snprintf (imname, 100, "/home/sathish/sourcecode/opecv_project/multitracker/inputframes/in%d.jpg",count);
  imwrite(imname, image);
  return blobrect;
}

// Fill the vector with random colors
void getRandomColors(vector<Scalar>& colors, int numColors)
{
  RNG rng(0);
  for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255))); 
}

int main(int argc, char* argv[])
{  
  VideoCapture vcap(1); 
  if(!vcap.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

        vcap.set(CV_CAP_PROP_FPS, 60);
        int fps = vcap.get(CV_CAP_PROP_FPS);
        cout<<"fps :"<<fps<<endl;
        cerr << "fps: " << fps<< endl;

  namedWindow("input", WINDOW_AUTOSIZE	);
  int frame_width=   vcap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height=   vcap.get(CV_CAP_PROP_FRAME_HEIGHT);
  cv::Rect ROI = Rect(0, 0, frame_width, frame_height);
  VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);

  //float expor = vcap.get(CV_CAP_PROP_EXPOSURE);
  //cout<<"expor :"<<expor<<endl;
  int count =0;

  Mat initframe;
  video.open("out.avi",CV_FOURCC('M','J','P','G'),30, Size(frame_width,frame_height),true);
  //(void)system("v4l2-ctl -d /dev/video1 -c exposure_auto=1"); //logitech
  //(void)system("v4l2-ctl -d /dev/video1 -c auto_exposure=1"); // ps3
  //(void)system("v4l2-ctl -d /dev/video1 -c exposure_absolute=10");
  //(void)system("v4l2-ctl -d /dev/video1 -c gain_automatic=0");
  //(void)system("v4l2-ctl -d /dev/video1 -c gain=0");
  (void)system("v4l2-ctl -d /dev/video1 -c auto_exposure=1");
  (void)system("v4l2-ctl -d /dev/video1 -c exposure=40");

	int frame_count = 0;
	int max_age = 1;
	int min_hits = 3;
	double iouThreshold = 0.3;
  vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq. 

	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	double cycle_time = 0.0;
	int64 start_time = 0;
  
  for(int fr=0;;fr++)
  {
    Mat frame;
    int init = 0;
    //(void)system("v4l2-ctl -d /dev/video1 -c exposure_absolute=10");  // logitech
    //(void)system("v4l2-ctl -d /dev/video1 -c exposure=10");  // PS3
    vcap >> frame;
    total_frames++;
		frame_count++;
    //video.write(frame);
    imshow( "input", frame );
    vector<cv::Rect> bboxes = {};
    cout<<"frame :"<<count<<endl;
    bboxes = getBoundingBoxs(frame, ROI, count);
    count++;
    if((bboxes.size() < 4) || (count < 10))
    {
      continue;
    }
    //vector<Scalar> colors;  
    //getRandomColors(colors, bboxes.size());

    
    if (trackers.size() == 0) // the first frame met
		{
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < bboxes.size(); i++)
			{
				KalmanTracker trk = KalmanTracker(bboxes[i]);
				trackers.push_back(trk);
        cout<<"kf init done for box  :"<<i<<"   "<<bboxes[i]<<endl;
			}
			continue;
		}

		///////////////////////////////////////
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();

		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
      cout<<"predictedBoxes  :"<<pBox<<endl;
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
        //cout<<"predictedBoxes 1 :"<<pBox<<endl;
				it++;
			}
			else
			{
				it = trackers.erase(it);
				cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}

		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		trkNum = predictedBoxes.size();
		detNum = bboxes.size();
    cout<<"trkNum :"<<trkNum<<endl;
    cout<<"detNum :"<<detNum<<endl;

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], bboxes[j]);
        cout<<"iouMatrix[i][j] :"<<iouMatrix[i][j]<<endl;
			}
		}


		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);
    for (int ass=0; ass<assignment.size();ass++)
    {
      cout<<"assignment :"<<assignment[ass]<<endl;
    }
		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();
    

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
      {
				allItems.insert(n);
        //cout<<"allItems :"<<allItems[n]<<endl;
      }

			for (unsigned int i = 0; i < trkNum; ++i)
      {
				matchedItems.insert(assignment[i]);
        //cout<<"allItems :"<<matchedItems[i]<<endl;
      }

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
    {
			;    
    }

    // filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

    for (cv::Point const& mp : matchedPairs)
    {
      cout <<"matched point :"<<mp << ' '<<endl;
    }

		///////////////////////////////////////
		// 3.3. updating trackers

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(bboxes[detIdx]);
      cout<<"trkIdx :"<<trkIdx<<endl;
      cout<<"detIdx :"<<detIdx<<endl;
      //cout<<"trackers[trkIdx]"<<trackers[trkIdx]<<endl;
		}    

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
      cout<<"umd :"<<bboxes[umd]<<endl;
			KalmanTracker tracker = KalmanTracker(bboxes[umd]);
			trackers.push_back(tracker);
		}

    // get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}


		if (1) // read image, draw results and show them
		{			
			for (auto tb : frameTrackingResult)
      {
				cv::rectangle(frame, tb.box, CV_RGB(255, 0, 0), 2, 8, 0);
        cout<<"tb.box :"<<tb.box<<endl;
      }
			imshow("MultiTracker", frame);
      video.write(frame);
			cvWaitKey(40);
		}  

    char c = (char)waitKey(33);
    if( c == 27 ) break;
  }
  return 0;
}

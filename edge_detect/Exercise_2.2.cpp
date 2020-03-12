#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

int main( )
{

  	Mat laps_dst, sobel_dst, bird, laps_abs_dst, sobel_abs_dst, canny_dst, canny_abs_dst;
	int ddepth = CV_16S;  // Depth of output image
	int scale = 1;		// Scale applied before assignment to dst
	int delta = 0;		// Offset applied before assignment to dst
	int ksize = 3;	// Kernel size

	/* Reading the inut image from commnd line */
	bird = imread( "./bird.jpg", -1 );
    if( bird.empty() ) return -1;
	/* Displaying the input image */
  	namedWindow( "bird", cv::WINDOW_AUTOSIZE );
  	imshow( "bird", bird );

	/* Apllying Laplacian edge detector on input image */
	Laplacian(bird, laps_dst, ddepth, ksize, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( laps_dst, laps_abs_dst );
	/* Displaaying and storing the laplacian edge map */
  	namedWindow( "Laplacian edge map", cv::WINDOW_AUTOSIZE );
  	imshow( "Laplacian edge map", laps_abs_dst );
	imwrite("./Laplacian_edge_map.jpg", laps_abs_dst);


	/* Applying sobel edge detector of 1st order derivates in x and y on input image */
	Sobel(bird, sobel_dst, ddepth, 1, 1, ksize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs( sobel_dst, sobel_abs_dst );
	/* Displaying and storing the sobel edge map */
  	namedWindow( "Sobel Edge Map", cv::WINDOW_AUTOSIZE );
  	imshow( "Sobel Edge Map", sobel_abs_dst );
	imwrite("./Sobel_Edge_Map.jpg", sobel_abs_dst);


	/* Applying cnny edge detector of 2:1 (high thereshold : low threshold) on input image */
	Canny(bird, canny_dst, 50, 100, ksize, true);
	convertScaleAbs( canny_dst, canny_abs_dst);
	/* Displaying and storing the sobel edge map */
  	namedWindow( "Canny Edge Map", cv::WINDOW_AUTOSIZE );
  	imshow( "Canny Edge Map", canny_abs_dst );
	imwrite("./Canny_Edge_Map.jpg", canny_abs_dst);

	cvWaitKey(0);

	destroyWindow( "bird" );
	destroyWindow( "Laplacian edge map" );
	destroyWindow( "Sobel Edge Map" );
	destroyWindow( "Canny Edge Map" );
}

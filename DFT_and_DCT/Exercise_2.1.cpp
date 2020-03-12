#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

int main( )
{

  	Mat input,idft_sav, idct_sav, idft_sav_l, idct_sav_l;

	/* Reading the input image from current directory */
	input = imread("./zebras.jpg", -1);
    if( input.empty() ) return -1;

	/* Displaying the input image */
  	namedWindow( "zebras", cv::WINDOW_AUTOSIZE );
  	imshow( "zebras", input);

	Mat input_float = cv::Mat::zeros( input.rows, input.cols, CV_32F );

	/* As the dft takes the float input, convert the input image to float by 
	   scaling with 1.0/255/.0 and store in input_float*/
	input.convertTo(input_float, input_float.type(), 1.0/255.0);

	//cout <<"Rows:" <<input.rows <<"coloums" <<input.cols << endl;

	/* get optimal size of the dft */

	int 	dft_rows = cv::getOptimalDFTSize( input.rows );
	int 	dft_cols = cv::getOptimalDFTSize( input.cols );

	//cout <<"Optimal Rows:" <<dft_rows <<"optimal coloums" <<dft_cols << endl;

	Mat dft_of_input = Mat::zeros( dft_rows, dft_cols, CV_32F );

	dft(input_float, dft_of_input, 0, input.rows );

	//imshow( "dft_of_input", dft_of_input);


	Mat idft_inputdft = Mat::zeros( dft_rows, dft_cols, CV_32F );

	dft( dft_of_input, idft_inputdft, DFT_INVERSE|DFT_SCALE, input.rows );

	imshow( "idft", idft_inputdft );
	idft_inputdft.convertTo(idft_sav, CV_8U, 255.0); 
	imwrite("./idft.jpg", idft_sav);

/*====================================================COMPUTING Lmin for DFT==========================================================*/

	/* L min = 1 for which we make all the pixels from 1<=K1<=255  and 1<=K2<=255 to 0 i.e we endup having only first row and coloum*/

	Mat Lmin = cv::Mat::zeros( input.rows, input.cols, CV_32F );	
	Mat dft_Lmin = cv::Mat::zeros( input.rows, input.cols, CV_32F );	 

	int i_dft, max_lim_dft,j_dft;
	i_dft = 52; // for L-min = 52 noticable degrade happended
	max_lim_dft = 255-i_dft+1;
	for (j_dft=0; j_dft<i_dft;j_dft++)
	{
		dft_of_input.row(j_dft).copyTo(Lmin.row(j_dft));
		dft_of_input.col(j_dft).copyTo(Lmin.col(j_dft));
	}
	for (i_dft=max_lim_dft; i_dft<256; i_dft++)
	{
		dft_of_input.row(i_dft).copyTo(Lmin.row(i_dft));
		dft_of_input.col(i_dft).copyTo(Lmin.col(i_dft));
	}

	Mat idft_Lmin = Mat::zeros( input.rows, input.cols, CV_32F );

	dft( Lmin, idft_Lmin, DFT_INVERSE|DFT_SCALE, input.rows );

	imshow( "idft For Lmin=52", idft_Lmin );
	idft_Lmin.convertTo(idft_sav_l, CV_8U, 255.0); 
	imwrite("./idft_for_Lmin=52.jpg", idft_sav_l );


/*========================================================DCT========================================================================*/


	Mat dct_of_input = cv::Mat::zeros( input.rows, input.cols, CV_32F );

	dct(input_float, dct_of_input);

	Mat idct_of_dct = Mat::zeros( input.rows, input.cols, CV_32F );

	dct( dct_of_input, idct_of_dct, DCT_INVERSE);


	imshow( "idct", idct_of_dct);
	idct_of_dct.convertTo(idct_sav, CV_8U, 255.0);
	imwrite("./idct.jpg", idct_sav);

/*====================================================COMPUTING Lmin for DCT==========================================================*/

	Mat Lmin2 = cv::Mat::zeros( input.rows, input.cols, CV_32F );	
	Mat dct_Lmin = cv::Mat::zeros( input.rows, input.cols, CV_32F );	 
	int j;  // FOr l-min = 70 noticable degrdation happened
	for (j=0; j<70; j++)
	{
		dct_of_input.row(j).copyTo(Lmin2.row(j));
		dct_of_input.col(j).copyTo(Lmin2.col(j));
	}

	Mat idct_Lmin = Mat::zeros( input.rows, input.cols, CV_32F );

	dct( Lmin2, idct_Lmin, DCT_INVERSE);

	imshow( "idct For L-min=70", idct_Lmin );
	idct_Lmin.convertTo(idct_sav_l, CV_8U, 255.0);
	imwrite("./idct_for_Lmin=70.jpg", idct_sav_l);



	cvWaitKey(0);

	destroyWindow( "zebras" );
	//destroyWindow( "dft_of_input" );
	destroyWindow( "idft" );
	//destroyWindow( "dft_of_Lmin" );
	destroyWindow( "idft For L-min=52" );
	destroyWindow( "idct" );
	destroyWindow( "idct For L-min=70" );

}

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <ctime>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
 
//Search Kernel for evaluating local gradient.
#define BLOCK_SIZE 24
//nxn block size for doing sum of absolute difference search.
//actual matching method is defined in the matchTemplate call
//to the opencv library.
#define SEARCH_SIZE 64
#define MAX_FEATURES 15
#define REQUESTED_IMAGE_HEIGHT 480
#define REQUESTED_IMAGE_WIDTH 640

//Undefine this value and run CVTest if you want to see
//how much cpu this is using.  All the debug output uses
//way more CPU than any of the real computations.
#define DEBUG

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace std::chrono;

typedef struct 
{
    uint16_t i;
    uint16_t j;
    float mG;
} ROI;

bool compareByGrad(const ROI &a, const ROI &b)
{
    return a.mG > b.mG;
}

ROI * rois;
ROI * disps;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, REQUESTED_IMAGE_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, REQUESTED_IMAGE_HEIGHT);

    vector<uchar> status;
    vector<float> err;

    Mat frame_tock;
    Mat grad;
    Mat debugImage;
    int inited = 0;
    int scale = 1;
    int delta = 0;
    int roiCount = 0;
    int dispsCount = 0;
    int actualWidth;
    int actualHeight;
    int ddepth = CV_16S;
    int tempy, tempx = 0;

    Mat grad_x, grad_y, grad_tick;
    Mat abs_grad_x, abs_grad_y;
    Mat absDiff;
    Mat result;
    Point minLoc;
    Point maxLoc;
    Scalar meanGrad;

    namedWindow("WIP",1);
    namedWindow("diff",2);
    for(;;)
    {
        Mat frame_left;
        Mat frame_right;
        cap >> frame_left; // get a new frame from camera
	debugImage = frame_left.clone();
        //resize(frame_left, frame_left, Size(REQUESTED_IMAGE_WIDTH, REQUESTED_IMAGE_HEIGHT), INTER_LINEAR); 

        clock_t begin = clock();

        if (inited == 0)
        {
            #ifdef DEBUG
            cout << "Width of output: " << frame_left.size().width << endl;
            cout << "Height of output: " << frame_left.size().height << endl;

            if ((frame_left.size().width != REQUESTED_IMAGE_WIDTH) || (frame_left.size().height != REQUESTED_IMAGE_HEIGHT))
            {
                cout << "WARNING: Capture device not outputting requested image size of: " << REQUESTED_IMAGE_WIDTH << " x " << REQUESTED_IMAGE_HEIGHT << endl;
            }
            #endif

            actualWidth = frame_left.size().width;
            actualHeight = frame_left.size().height;

            rois = (ROI *) calloc((actualHeight * actualWidth) / BLOCK_SIZE, sizeof(ROI));
            //disps = (ROI *) calloc((actualHeight * actualWidth) / BLOCK_SIZE, sizeof(ROI));
        }                 

        #ifdef DEBUG
        //absdiff(frame_left, frame_tock, debugImage);
        #endif  

        cvtColor(frame_left, frame_left, COLOR_BGR2GRAY);

        if (!inited)
        {
            frame_tock = frame_left;
            inited = 1;
        }

#ifdef DEBUG
        Mat debugImageDisp(actualHeight, actualWidth, CV_8UC3, Scalar(255,255,255));
#endif
        Mat left_grad, right_grad;

        //GaussianBlur( frame_left, frame_left, Size(3,3), 0, 0, BORDER_DEFAULT );
        //Laplacian( frame_left, left_grad, CV_16S, 3, 1, 0, BORDER_DEFAULT );
        //convertScaleAbs( left_grad, grad_tick );

        grad_tick = frame_left;

        #ifdef DEBUG
        //cv::cvtColor(grad_tick, debugImageL, CV_GRAY2RGB);
        #endif

        int i; //ROW
        int j; //COLUMN
        int k;
        
        float x_av = 0;
        float y_av = 0;
        float disp = 0;
        uint64_t count = 0;
	static uint8_t cooldown = 0;

        roiCount = 0;
        dispsCount = 0;

        //Find all areas of interest by gradient 
        for (i = (BLOCK_SIZE*2); i < (actualWidth - SEARCH_SIZE); i += BLOCK_SIZE)
        {
            for (j = (BLOCK_SIZE) + 2; j < (actualHeight - ((BLOCK_SIZE*2) + 2)); j += BLOCK_SIZE)
            {
                /*
                if (((i - ((SEARCH_SIZE) / 2)) < 0) || (j - ((BLOCK_SIZE) / 2) < 0) || ((i + SEARCH_SIZE) > actualWidth) || ((j + BLOCK_SIZE) > actualHeight))
                {
                    continue;
                }
                */

                //Mat Rec(i, j, 8, 8);
                meanGrad = mean(grad_tick(Rect(i, j, BLOCK_SIZE, BLOCK_SIZE)));

                //ixj
                rois[roiCount].i = i;
                rois[roiCount].j = j;
                rois[roiCount].mG = (float) meanGrad[0];

                roiCount++;
            }
        }

	float X = 0;
	float Y = 0;
	int num = 0;
        
        //TODO: Make the sort faster and do it on the fly in the previous loop.
        //std::sort(&rois[0], &rois[roiCount], compareByGrad);

        for (k = 0; k < MAX_FEATURES && (k < roiCount); k++)
        {
            //Use the strongest features, but also set a lower limit.
            //if (rois[k].mG < 1)
            //{
            //    continue;
            //}

            absdiff(frame_tock(Rect(rois[k].i, rois[k].j, BLOCK_SIZE, BLOCK_SIZE)), frame_left(Rect(rois[k].i, rois[k].j, BLOCK_SIZE, BLOCK_SIZE)), absDiff);

            Scalar meanDiff = mean(absDiff);
            
            //Origin of the search window.
            int sXi = rois[k].i - (SEARCH_SIZE/2) + (BLOCK_SIZE/2);
            int sYi = rois[k].j - (SEARCH_SIZE/2) + (BLOCK_SIZE/2);

            bool test = false;

            if ((meanDiff[0] > 20) && sXi >= 0 && sYi >=0 && (sXi + SEARCH_SIZE) < actualWidth && (sYi + SEARCH_SIZE) < actualHeight)
            {
                test = true;  
            }

		std::ostringstream str;

            if (test == true)
            {
                //Calculate L1 norm from last frame's window to the current one.
                //SAD is very parallelizable, see if we can SIMD/NEON this on the A9...
                
                matchTemplate(
                    frame_left(Rect(sXi, sYi, SEARCH_SIZE, SEARCH_SIZE)),
                    frame_tock(Rect(rois[k].i, rois[k].j, BLOCK_SIZE, BLOCK_SIZE)), result, CV_TM_CCORR_NORMED
                    );//CV_TM_SQDIFF_NORMED);
                //matchTemplate(frame_left(Rect(i - ((SEARCH_SIZE - BLOCK_SIZE) / 2), j - ((SEARCH_SIZE - BLOCK_SIZE) / 2), SEARCH_SIZE, SEARCH_SIZE)), frame_tock(Rect(i, j, BLOCK_SIZE, BLOCK_SIZE)), result, CV_TM_CCORR);
                normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());
                minMaxLoc(result, NULL, NULL, &minLoc, &maxLoc);        

                #ifdef DEBUG
                //int tempx;
                //int tempy;

                tempx = maxLoc.x;
                tempy = maxLoc.y;

		X += (float) ( tempx - ((SEARCH_SIZE-BLOCK_SIZE)/2));
		Y += (float) ( tempy - ((SEARCH_SIZE-BLOCK_SIZE)/2));

		num++;

                rectangle(debugImage, Rect(rois[k].i, rois[k].j, BLOCK_SIZE, BLOCK_SIZE), Scalar(0, 0, 100), 1, 8, 0);
                rectangle(debugImage, Rect(sXi, sYi, SEARCH_SIZE, SEARCH_SIZE), Scalar(0, 30, 0), 1, 8, 0);
                arrowedLine(debugImage, Point(rois[k].i + (BLOCK_SIZE/2), rois[k].j + (BLOCK_SIZE/2)), Point(sXi + tempx + (BLOCK_SIZE/2), sYi + tempy + (BLOCK_SIZE/2)), Scalar(0, 200, 0), 1, 8, 0); 
                #endif
            }
        }

	static uint8_t saveImageThreshold = 0;

	if (num > 0)
	{
		arrowedLine(debugImage, Point(actualWidth/2, actualHeight/2),  Point((actualWidth/2) + (X/num), (actualHeight/2) + (Y/num)), Scalar(200, 200, 200), 1, 8, 0);
	}

	if (num >= 4)
	{
		saveImageThreshold++;
	}
	else
	{
		saveImageThreshold = 0;
	}
	
	if (cooldown != 0)
	{
		cooldown--;
	}
	else if (saveImageThreshold >= 3)
	{	
        	std::ostringstream str;

		time_t result = std::time(nullptr);
		
		if (X/num < 0)
		{
			str << "bb_" << std::asctime(std::localtime(&result)) << ".jpg";
		}
		else
		{
			str << std::asctime(std::localtime(&result)) << ".jpg";
		}

		imwrite(str.str(), debugImage);
		saveImageThreshold = 0;
		cooldown = 35;
	}


        #ifdef DEBUG
        clock_t end = clock();
        double elapsed_msecs = double(end - begin) / CLOCKS_PER_SEC * 1000;

        cout << "loop time: " << elapsed_msecs << " mS" << endl;
        x_av = y_av = count = 0;
        #endif

	absdiff(frame_tock, frame_left, absDiff);
        frame_tock = frame_left;


#ifdef DEBUG
        //-- Show detected (drawn) keypoints
        //imshow("WIP", debugImage );
	//imshow("Diff", absDiff);
        if(waitKey(1) >= 0) break;
#endif
        
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

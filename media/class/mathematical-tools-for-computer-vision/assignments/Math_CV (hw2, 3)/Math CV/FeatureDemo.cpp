// FeatureDemo.cpp : Defines the entry point for the console application.
//



#include <stdio.h>
#include <tchar.h>
#include <SDKDDKVer.h>

#include "windows.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#ifdef _DEBUG
#pragma comment(lib,"opencv_world300d.lib")
#else
#pragma comment(lib,"opencv_world300.lib")
#endif

#include "PolygonDemo.h"

using namespace std;
using namespace cv;

void onMouse(int evt, int x, int y, int flags, void* param)
{
    PolygonDemo *p = (PolygonDemo *)param;
    p->HandleEvent(evt, x, y, flags);
}

int _tmain(int argc, _TCHAR* argv[])
{
    // main image
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    imshow("PolygonDemo", frame);

    // event handler µî·Ï
    PolygonDemo tmp;
    setMouseCallback("PolygonDemo", onMouse, &tmp);

    char ch = waitKey();
    if (ch == 27) return 0;				// ESC Key
	return 0;
}


#include "PolygonDemo.h"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
    m_polygon = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::RefreshImage()
{
    Mat frame = Mat::zeros(480, 640, CV_8UC3);

    DrawPolygon(frame, m_vtx, m_polygon);
    if (m_polygon)
    {
        // pt in polygon
        for (int i = 0; i < (int)m_pts.size(); i++)
        {
            if (PtInPolygon(m_vtx, m_pts[i]))
            {
                circle(frame, m_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
            }
            else
            {
                circle(frame, m_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
            }
        }

        // polygon area
        int area = PolyArea(m_vtx);
        char str[100];
        sprintf_s(str, 100, "Area = %d", area);
        putText(frame, str, Point(115, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0,255,255), 1);

        // homography check
        if (m_vtx.size() == 4)
        {
            // rect points
            int rect_sz = 100;
            vector<Point> rc_pts;
            rc_pts.push_back(Point(0, 0));
            rc_pts.push_back(Point(0, rect_sz));
            rc_pts.push_back(Point(rect_sz, rect_sz));
            rc_pts.push_back(Point(rect_sz, 0));
            rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

            // draw mapping
            char* abcd[4] = { "A", "B", "C", "D" };
            for (int i = 0; i < 4; i++)
            {
                line(frame, rc_pts[i], m_vtx[i], Scalar(255, 0, 0), 1);
                circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                circle(frame, m_vtx[i], 2, Scalar(0, 255, 0), CV_FILLED);
                putText(frame, abcd[i], m_vtx[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
            }

            // check homography
            int homo_type = ClassifyHomography(rc_pts, m_vtx);
            char type_str[100];
            switch (homo_type)
            {
            case NORMAL:
                sprintf_s(type_str, 100, "normal");
                break;
            case CONCAVE:
                sprintf_s(type_str, 100, "concave");
                break;
            case TWIST:
                sprintf_s(type_str, 100, "twist");
                break;
            case REFLECTION:
                sprintf_s(type_str, 100, "reflection");
                break;
            case CONCAVE_REFLECTION:
                sprintf_s(type_str, 100, "concave reflection");
               break;
            }

            putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
        }
    }

    imshow("PolygonDemo", frame);
}

// 외적을 이용하여 면적 구하기
int PolygonDemo::PolyArea(const std::vector<cv::Point>& vtx)
{
	int sum = 0;
	int signed_area = 0;
	int size = (int)m_vtx.size();

	for (int i = 1; i < size-1; i++) {
		
		signed_area = ((vtx[i].x - vtx[0].x)*(vtx[i+1].y - vtx[0].y) - (vtx[i].y - vtx[0].y)*(vtx[i+1].x - vtx[0].x))/2;
		sum = sum + signed_area;
	}

	if (sum < 0 )	sum = (-1)*sum;
	return sum;
}

// return true if pt is interior point
bool PolygonDemo::PtInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{



    return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::ClassifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
    if (pts1.size() != 4 || pts2.size() != 4) return -1;

	bool base_sign[4] = {true};
	bool target_sign[4] = {true};

	for (int i=0; i<4; i++) {
		base_sign[i] = sign_crossproduct(pts1, down_num(i), i, up_num(i));
		target_sign[i] = sign_crossproduct(pts2, down_num(i), i, up_num(i));
	}

	int count =0;
	for (int i=0; i<4; i++) {

		
		if( target_sign[i] == base_sign[i] ) {
			count++;
		}
	}

	if(count == 4)
		return NORMAL;
	if(count == 2)
		return TWIST;
	if(count == 0)
		return REFLECTION;
	else
		return CONCAVE;
}


int PolygonDemo::down_num(int a)
{
	if(a==0) return 3;
	else
		return --a;
}

int PolygonDemo::up_num(int a)
{
	if(a==3) return 0;
	else
		return ++a;
}


bool PolygonDemo::sign_crossproduct(const std::vector<cv::Point>& vtx, int im, int i, int ip)
{
	int result;
	
	result = (vtx[i].x - vtx[im].x)*(vtx[ip].y - vtx[im].y) - (vtx[i].y - vtx[im].y)*(vtx[ip].x - vtx[im].x);
	


	if(result > 0)
		return true;
	else
		return false;
}



void PolygonDemo::DrawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
    int i = 0;
    for (i = 0; i < (int)m_vtx.size(); i++)
    {
        circle(frame, m_vtx[i], 2, Scalar(255, 255, 255), CV_FILLED);
    }
    for (i = 0; i < (int)m_vtx.size() - 1; i++)
    {
        line(frame, m_vtx[i], m_vtx[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        line(frame, m_vtx[i], m_vtx[0], Scalar(255, 255, 255), 1);
    }
}

void PolygonDemo::HandleEvent(int evt, int x, int y, int flags)
{
    if (evt == CV_EVENT_LBUTTONDOWN)
    {
        if (!m_polygon)
        {
            m_vtx.push_back(Point(x, y));
        }
        else
        {
            m_pts.push_back(Point(x, y));
        }
        RefreshImage();
    }
    else if (evt == CV_EVENT_LBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_LBUTTONDBLCLK)
    {
        m_polygon = true;
        RefreshImage();
    }
    else if (evt == CV_EVENT_RBUTTONDBLCLK)
    {
    }
    else if (evt == CV_EVENT_MOUSEMOVE)
    {
    }
    else if (evt == CV_EVENT_RBUTTONDOWN)
    {
        m_vtx.clear();
        m_pts.clear();
        m_polygon = false;
        RefreshImage();
    }
    else if (evt == CV_EVENT_RBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_MBUTTONDOWN)
    {
    }
    else if (evt == CV_EVENT_MBUTTONUP)
    {
    }

    if (flags&CV_EVENT_FLAG_LBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_RBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_MBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_CTRLKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_SHIFTKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_ALTKEY)
    {
    }
}

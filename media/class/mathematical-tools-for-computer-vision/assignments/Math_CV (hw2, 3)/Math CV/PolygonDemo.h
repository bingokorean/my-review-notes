#pragma once

#include "opencv2/highgui.hpp"


class PolygonDemo
{
public:
    PolygonDemo();
    ~PolygonDemo();

    void RefreshImage();
    void HandleEvent(int evt, int x, int y, int flags);
    void DrawPolygon(cv::Mat& frame, const std::vector<cv::Point>& vtx, bool closed);

    bool PtInPolygon(const std::vector<cv::Point>& vtx, cv::Point pt);
    int PolyArea(const std::vector<cv::Point>& vtx);

    enum {NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION};
    int ClassifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2);

	int up_num(int a);
	int down_num(int a);
	bool sign_crossproduct(const std::vector<cv::Point>& pts, int im, int i, int ip);
protected:
    bool m_polygon;
    std::vector<cv::Point> m_vtx;
    std::vector<cv::Point> m_pts;
};


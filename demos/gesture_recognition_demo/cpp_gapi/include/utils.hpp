#include <opencv2/core.hpp>

struct Detection
{
    int id;
    cv::Rect roi;
    float conf;
    int waiting;
    int duration;
};


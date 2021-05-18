#include <opencv2/core.hpp>

struct rectHash{
    std::size_t operator()(const cv::Rect& rect) const {
        std::hash<int> hashVal;
        return hashVal(static_cast<int>(floor(rect.x / rect.width) + floor(rect.y / rect.height)));
  }
};

struct rectEqual{
    bool operator() (const cv::Rect& l, const cv::Rect& r) const {
        return l.x == r.x && l.y == r.y &&
               l.width == r.width &&
               l.height == r.height;
    }
};

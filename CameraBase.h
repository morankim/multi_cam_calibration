#include <opencv2/opencv.hpp>

class CameraBase
{
private:
    /* data */
public:
    CameraBase(int w, int h) {width = w; height = h;}
    ~CameraBase() {}

    int width;
    int height;
    int nParams = 2;

    void undistort();
};


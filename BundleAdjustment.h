#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres.h>
#include <opencv2/opencv.hpp>
#include <Eigen/LU>
#include <Eigen/Dense>
using namespace std;
using namespace cv;

#if 1
typedef std::vector<cv::Point2f> corners_per_img;
typedef std::vector<int> idx_per_img;
typedef std::vector<idx_per_img> idx_per_cam;
typedef std::vector<corners_per_img> corners_per_cam;
typedef std::vector<int> visibility_per_img;
typedef std::vector<visibility_per_img> visibility_per_cam;
#endif

struct Point3DInMap
{
    cv::Point3f p;
    std::map<int, int> originatingViews;//what is it?
};

namespace sfmlib{

    struct Intrinsics
    {
        double focal_x, focal_y;
        double k1, k2;
        double princ_x, princ_y;
    };

    class SparseBundleAdjustment
    {
    public:
        SparseBundleAdjustment() {}

        vector<Point3DInMap> pointCloud;
        Intrinsics intrin;
        vector<double> world3d;
        //n_cams, n_images, n_points
        vector<vector<vector<cv::Point2f>>> image2d;

        template<typename T>
        void eulerAnglesToRotationMatrix(const T* const extr, T** rota_mat);
        template<typename T>
        void invert_rotation_matrix(T** rota_mat);
        template<class iter1, class iter2, class out>
        static out find_intersection(iter1 s1, iter1 e1,
                     iter2 s2, iter2 e2,
                     out result, vector<Vec2i>& common_id)
        {
            // s1, s2 are already ordered vectors
            int cnt1=0;
            int cnt2=0;
            while(s1!=e1 && s2!=e2)
            {
                if(*s1<*s2) {++s1; cnt1++;}
                else if(*s2<*s1) {++s2; cnt2++;}
                else
                {
                    *result = *s1;
                    common_id.push_back(Vec2i(cnt1, cnt2)) ;
                    ++result; ++s1; ++s2;
                    cnt1++; cnt2++;
                }
            }
            return result;
        }

        static void adjustBundle(vector<Intrinsics>& intrins,
            const vector<cv::Point3f>& world3d,
            const vector<corners_per_cam> mv_charucoCorners,
            const vector<idx_per_cam> multi_view_charuoIdx,
            const vector<vector<cv::Mat>>& R_pattern2cam,
            const vector<vector<cv::Mat>>& T_pattern2cam,
            const vector<cv::Mat>& camPoses_mat,
                  vector<cv::Mat>& camPoses_vec,
                  vector<vector<cv::Mat>>& patPoses_vec);
    };
}



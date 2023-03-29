//#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include "BundleAdjustment.h"

using namespace std;
typedef vector<vector<cv::Point2f>> point2D_vector;

int squaresX = 6; 
int squaresY = 9;
int numTotalCorners = (squaresX-1)*(squaresY-1);
int max_num_corners_on_a_line = std::max(squaresX, squaresY);
float squareLength = 0.021;//unit: meter 90;//0.04; //(normally in meters)
float markerLength = 0.012;//50;//0.02; //(normally in meters)
int margins = 10; // pixel unit?
int dictionaryId = 10;
const Ptr<cv::aruco::Dictionary> &dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
cv::Ptr<cv::aruco::CharucoBoard> charBoard = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
cv::Ptr<cv::aruco::Board> board = charBoard.staticCast<cv::aruco::Board>();
cv::Ptr<cv::aruco::DetectorParameters> detectorParameters = cv::aruco::DetectorParameters::create();
cv::Size img_size;
int min_number_of_pts_detected = 6;//std::max(squaresX, squaresY)-1;

// single camera calibration class
class CameraSystemCalibration
{
private:
    /* data */
public:
    CameraSystemCalibration(/* args */);
    ~CameraSystemCalibration();

    // given object points and image points, outputs K, Rt matrix
    // RandomPatternCornerFinder- is that working right? I'm not sure

    void init_pattern_board(const cv::Size& imgSize)
    {
        // setting 2d image points
        img_size = imgSize;
        charBoard->draw(img_size, charuco_board, margins);
        cv::imwrite("charuco_pattern_9x6_1.png", charuco_board);

        // setting 3d object points/ fill the chessboard corner
        objPts.clear();
        for(int y=0; y<squaresY-1; y++){
            for(int x=0; x<squaresX-1; x++){
                cv::Point3f corner;
                corner.x = (x+1)*squareLength;
                corner.y = (y+1)*squareLength;
                corner.z = 0;
                objPts.push_back(corner);
            } // vert, hori size check
        }
        
    }

    void multiCamCalibration(vector<sfmlib::Intrinsics>& intrins,
            const vector<cv::Point3f>& world3d,
            const vector<corners_per_cam> filled_mv_charucoCorners,
            const vector<idx_per_cam> mv_charuoIdx,
                  vector<vector<cv::Mat>>& Rvec_pattern2cam,
                  vector<vector<cv::Mat>>& Tvec_pattern2cam,
                  vector<cv::Mat>& camPoses_mat,
                  vector<cv::Mat>& camPoses_vec,
                  vector<vector<cv::Mat>>& patPoses_vec)
    {
        cout<<"//---------------multi cam calibration start! ---------------//"<<endl;
        sfmlib::SparseBundleAdjustment::adjustBundle(intrins, world3d, filled_mv_charucoCorners, mv_charuoIdx, 
                                                     Rvec_pattern2cam, Tvec_pattern2cam, camPoses_mat, camPoses_vec, patPoses_vec);
    }

    void apply_Ransac_to_corresp2D3D(const float& thres, corners_per_img& charucoCorners, idx_per_img& charucoIdx)
    {
        float thres_square = thres*thres;
        vector<cv::Point2f> worldPts;
        worldPts.resize(charucoIdx.size());

        for(int i=0; i<charucoIdx.size(); i++)
        {
           int id = charucoIdx[i];
           worldPts[i] = cv::Point2f(objPts[id].x, objPts[id].y);
        }

        cv::Mat homo = cv::findHomography(worldPts, charucoCorners, cv::RANSAC, 5.0);
        homo.convertTo(homo, CV_32F);
        if(homo.dims < 1){
            // if empty matrix
            charucoCorners.clear();
            charucoIdx.clear();
        }

        else{
        for(int i=0; i<charucoIdx.size(); i++)
        {

            float HX_x      =      homo.at<float>(0, 0)*worldPts[i].x + homo.at<float>(0, 1)*worldPts[i].y + homo.at<float>(0, 2);
            float HX_y      =      homo.at<float>(1, 0)*worldPts[i].x + homo.at<float>(1, 1)*worldPts[i].y + homo.at<float>(1, 2);
            float scale_inv = 1.0/(homo.at<float>(2, 0)*worldPts[i].x + homo.at<float>(2, 1)*worldPts[i].y + homo.at<float>(2, 2));
            cv::Point2f projPt;
            projPt.x = HX_x*scale_inv;
            projPt.y = HX_y*scale_inv;
            float err = (charucoCorners[i].x-projPt.x)*(charucoCorners[i].x-projPt.x)+(charucoCorners[i].y-projPt.y)*(charucoCorners[i].y-projPt.y);
            if(err>thres_square)
            {
                worldPts.erase(worldPts.begin()+ i);
                charucoCorners.erase(charucoCorners.begin()+i);
                charucoIdx.erase(charucoIdx.begin()+i);
                i--;
            }
        }}
    }

    // Assume we are using chessboard , single camera calibration
    //---------------------------------------------------------------------------------------------------//
    double CamCalibration(corners_per_cam& charucoCorners, idx_per_cam& charucoIdx,
                          cv::Mat& K, cv::Mat& dist, vector<cv::Mat>& rvec, vector<cv::Mat>& tvec)
    {
        rvec.clear();
        tvec.clear();
        // for every captured pattern images, find image corners
        K = cv::Mat::eye(3, 3, CV_64F);
        double reproErr = cv::aruco::calibrateCameraCharuco(charucoCorners, charucoIdx, charBoard, img_size, K, dist, rvec, tvec, CALIB_FIX_PRINCIPAL_POINT);
        return reproErr;
    }

    bool detect_unstable_corners(const vector<int>& charucoIdx)
    {
        int max_corners_xy = std::max(squaresX, squaresY);
        bool on_a_line_xaxis = true;
        bool on_a_line_yaxis = true;

        int prev_id = charucoIdx[0];
        int next_id;
        for(int i = 1; i < charucoIdx.size(); i++){
            next_id = charucoIdx[i]%(squaresX-1);
            if(next_id != prev_id){
                on_a_line_xaxis = false; 
                break;
            }
            else prev_id = next_id;
        }
        prev_id = charucoIdx[0];
        for(int i=1; i<charucoIdx.size(); i++){
            next_id = charucoIdx[i]%(squaresY-1);
            if(next_id != prev_id)
            {
                on_a_line_yaxis = false; 
                break;
            }
        }

        if(on_a_line_xaxis==true || on_a_line_yaxis==true) return true;
        else return false;
    }

    //---------------------------------------------------------------------------------------------------//
    void detect_charuco_markers(const int& cam_id, const vector<cv::Mat>& imgs, corners_per_cam& charucoCorners, 
                                idx_per_cam& charucoIdx, visibility_per_img& how_many_pts_seen)
    {
        int num_whole_imgs = imgs.size();
        vector<vector<cv::Point2f>> rejected;
        vector<vector<cv::Point2f>> markerCorners;
        vector<int> markerIdx;
        charucoCorners.clear();
        charucoIdx.clear();
        how_many_pts_seen.clear(); how_many_pts_seen.resize(num_whole_imgs, -11);

        // detect markers
        cout<<"#charCorners: ";
        for(int i = 0; i < num_whole_imgs; i++)
        {
            markerCorners.clear();
            markerIdx.clear();
            rejected.clear();
            cv::aruco::detectMarkers(imgs[i], dictionary, markerCorners, markerIdx, detectorParameters, rejected);
            if(markerIdx.size() < 1) continue;
            cv::aruco::refineDetectedMarkers(imgs[i], board, markerCorners, markerIdx, rejected);
            // interpolate charuco corners
            corners_per_img charCorners_;
            idx_per_img charIdx_;
            cv::aruco::interpolateCornersCharuco(markerCorners, markerIdx, imgs[i], charBoard, charCorners_, charIdx_);
            if(charIdx_.size() < min_number_of_pts_detected) continue;

            apply_Ransac_to_corresp2D3D(4.0, charCorners_, charIdx_);
            if(charIdx_.size() < 5) continue;

            charucoCorners.push_back(charCorners_);
            charucoIdx.push_back(charIdx_);
            // update visibility information for each view
            int num_detected_markers = charIdx_.size();
            how_many_pts_seen[i] = num_detected_markers;
            //cout<<"["<<i<<" th img, "<<charucoIdx.size()-1<<", "<<charCorners_.size()<<"] ";

            // draw results
            bool draw = true;
            if(draw){
                cv::Mat copy_img;
                imgs[i].copyTo(copy_img);
                if(markerIdx.size()>0) cv::aruco::drawDetectedMarkers(copy_img, markerCorners);
                if(charIdx_.size()>0) cv::aruco::drawDetectedCornersCharuco(copy_img, charCorners_, charIdx_);
                cv::imwrite(cv::format("%d/detected_aruco_corners%02d.png", cam_id, i), copy_img);
            }
        }
    }

    cv::Mat charuco_board;
    vector<cv::Point3f> objPts;
    
    int minMatchedPoints = 20;
    float maxInitReprojectionError = 5.0;
    float maxInlierError;
    float maxSmoothError;
    vector<double> intrinsics;

    cv::Mat pattern;
    int patternPoints;
    int patternFeatuers;
};

CameraSystemCalibration::CameraSystemCalibration(/* args */)
{
}

CameraSystemCalibration::~CameraSystemCalibration()
{
}

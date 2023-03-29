// Random pattern calibration for multiple-camera system
// it first calibrates each camera individually, then a bundle adjustment
// optimization is applied to refine extrinsic parameters
// it only support "random" pattern for calibration. 
// camIdx must start from 0

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <omp.h>
#include "CameraSystemCalibration.h"
#include <VACS.h>

using namespace std;
using namespace cv;
using namespace VACS;
typedef vector<vector<cv::Point2f>> point2D_vector;

int readImagesFromVideo(const string filename, const string outFolderName, const int camIdx, const float scale, vector<cv::Mat>& imgs, int& n);
cv::Mat computeRelativeRt(const cv::Mat& rvec1, const cv::Mat& t1,const cv::Mat& rvec2, const cv::Mat& t2);
cv::Mat downsize_img(const cv::Mat& img, const float downSizeScale);
void loadImages(const string folderName, vector<cv::Mat>& imgs);
void find_common_scene_index_for_twoView(const vector<int>& v1, const vector<int>& v2, cv::Point2i& idx);
void calibQuality_after(const sfmlib::Intrinsics& intrins,
            const vector<cv::Point3f>& world3d,
            const corners_per_cam& mv_charucoCorners,
            const idx_per_cam& multi_view_charuoIdx,
            const cv::Mat& camPoses_vec,
            const vector<cv::Mat>& patPoses_vec_ref,
            const visibility_per_img& vis_ref,
            const visibility_per_img& vis);

void calibQuality_before(const sfmlib::Intrinsics& intrins,
            const vector<cv::Point3f>& world3d,
                 corners_per_cam& mv_charucoCorners,
                 idx_per_cam& multi_view_charuoIdx,
            const visibility_per_img& vis,
            const vector<cv::Mat>& Rvec_pattern2cam,
            const vector<cv::Mat>& Tvec_pattern2cam,
            const float& reprojErr_thres,
            const vector<cv::Mat>& pattern_imgs);

string extractIntegerWords(const string str);
void check_relative_transf(const sfmlib::Intrinsics& Kl, const sfmlib::Intrinsics& Kr, const vector<cv::Point3f>& world3d, 
    const corners_per_img& pl, const idx_per_img& id_l, const corners_per_img& pr, const idx_per_img& id_r, const cv::Mat& camPoses_vec, cv::Mat& patPose_l, cv::Mat& patPose_r);

void fill_empty_space(const int& numTotalCorners, const visibility_per_img& vis, const idx_per_cam& mv_charuoIdx, 
    const corners_per_cam& mv_charucoCorners, corners_per_cam& filled_mv_charucoCorners);

void tri_reproErr(const int& nWholeCorners, const sfmlib::Intrinsics& lcam, const sfmlib::Intrinsics& rcam, 
    const corners_per_img& pl, const idx_per_img& id_l, const corners_per_img& pr, const idx_per_img& id_r, cv::Mat& camPoses_vec);
void save_calib_result(const string& filename, const vector<sfmlib::Intrinsics>& Ks, const vector<cv::Mat>& camPose_vec, const cv::Mat& relpose_mesh2refcam);
void load_user_input(vector<cv::Point2f>& left_corresp_pts, vector<cv::Point2f>& right_corresp_pts)
{
    left_corresp_pts.clear();
    left_corresp_pts.push_back(cv::Point2f(1949, 777));
    left_corresp_pts.push_back(cv::Point2f(1989, 792));
    left_corresp_pts.push_back(cv::Point2f(2337, 762));
    left_corresp_pts.push_back(cv::Point2f(1985, 1420));
    left_corresp_pts.push_back(cv::Point2f(2564, 1521));

    right_corresp_pts.clear();
    right_corresp_pts.push_back(cv::Point2f(1732, 953));
    right_corresp_pts.push_back(cv::Point2f(1764, 967));
    right_corresp_pts.push_back(cv::Point2f(2123, 943));
    right_corresp_pts.push_back(cv::Point2f(1713, 1598));
    right_corresp_pts.push_back(cv::Point2f(2312, 1720));
}

cv::Mat estimateRT_btw_mesh_n_refCam(const sfmlib::Intrinsics& Kl, const sfmlib::Intrinsics& Kr, cv::Mat& camPoses_l, cv::Mat& camPoses_r,
    const vector<cv::Point2f>& lpts, const vector<cv::Point2f>& rpts);

int main(int argc, char* argv[] )
{
    // setting input folder names
    vector<string> fname1;
    //fname1.push_back("20190430/A");
    //fname1.push_back("20190430/B");
    //fname1.push_back("20190430/C");
    fname1.push_back("20190619/A");
    fname1.push_back("20190619/B");
    fname1.push_back("20190619/C");
    fname1.push_back("20190619/D");
    fname1.push_back("20190619/E");
    fname1.push_back("20190619/F");
    fname1.push_back("20190619/G");

    int numCams = fname1.size();
    vector<vector<cv::Mat>> gray_pattern_images;
    gray_pattern_images.resize(numCams);

    for(int i=0; i<numCams; i++){
        loadImages(fname1[i], gray_pattern_images[i]);
    }
    cout<<"All images are saved!."<<endl;

    // calibrate each camera
    // CameraSystemCalibration calib;
    vector<sfmlib::Intrinsics> Ks; Ks.resize(numCams);
    vector<vector<cv::Mat>> Rvecs; Rvecs.resize(numCams);
    vector<vector<cv::Mat>> tvecs; tvecs.resize(numCams);
    vector<corners_per_cam> mv_charucoCorners; mv_charucoCorners.resize(numCams);
    vector<corners_per_cam> filled_mv_charucoCorners; filled_mv_charucoCorners.resize(numCams);
    vector<idx_per_cam> mv_charuoIdx; mv_charuoIdx.resize(numCams);
    visibility_per_cam vis_2view; vis_2view.resize(numCams);
    vector<cv::Mat> camPoses_vec; camPoses_vec.resize(numCams);
    vector<cv::Mat> camPoses_mat; camPoses_mat.resize(numCams);
    vector<vector<cv::Mat>> patPoses_vec;
    vector<cv::Point2i> common_scene_index; common_scene_index.resize(numCams-1);

    CameraSystemCalibration calib;
    calib.init_pattern_board(gray_pattern_images[0][0].size());
    omp_set_num_threads(8);
    #pragma opm parallel for
    for(int i=0; i<numCams; i++)
    {
        cout<<"------------ "<<i<<" th cam calibration start. ------------"<<endl;
        cv::Mat K, Kd;
        calib.detect_charuco_markers(i, gray_pattern_images[i], mv_charucoCorners[i], mv_charuoIdx[i], vis_2view[i]);
        double reprojErr = calib.CamCalibration(mv_charucoCorners[i], mv_charuoIdx[i], K, Kd, Rvecs[i], tvecs[i]);
        cout<<"corners size: "<<mv_charucoCorners[i].size()<<", Rvecs size: "<<Rvecs[i].size()<<endl;

        sfmlib::Intrinsics intrinsic; // intrinsics from quarter size of the original image
        intrinsic.focal_x = K.at<double>(0, 0); 
        intrinsic.focal_y = K.at<double>(1, 1);
        intrinsic.princ_x = K.at<double>(0, 2);
        intrinsic.princ_y = K.at<double>(1, 2);
        intrinsic.k1 = Kd.at<double>(0);
        intrinsic.k2 = Kd.at<double>(1);
        Ks[i]= intrinsic;
        cout<<mv_charucoCorners[i].size()<<" number of images are used."<<endl;
        cout<<intrinsic.focal_x<<", "<<intrinsic.focal_y<<", "<<intrinsic.princ_x<<", "<<intrinsic.princ_y<<", "<<intrinsic.k1<<", "<<intrinsic.k2<<endl;
        cout<<"--------------  reprojErr: "<<reprojErr<<"  --------------"<<endl;
    }

    // find image index with whole number of corners are captured from consecutive cameras
    for(int i=0; i<numCams-1; i++){
        find_common_scene_index_for_twoView(vis_2view[i], vis_2view[i+1], common_scene_index[i]);
    }
    
    // set relative transformation for each cameras
    vector<cv::Mat> rel_pose_btw2;
    rel_pose_btw2.resize(numCams);
    rel_pose_btw2[0] = cv::Mat::eye(4, 4, CV_64F);
    vector<cv::Point2i> common_view_with_maxPts; common_view_with_maxPts.resize(numCams-1);
    for(int i=1; i<numCams; i++)
    {
        // nth image which contains many features
        int left_id  = common_scene_index[i-1].x; 
        int right_id = common_scene_index[i-1].y; 
        // relative Rt vector independent coordinate
        rel_pose_btw2[i] = computeRelativeRt(Rvecs[i-1][left_id], tvecs[i-1][left_id], Rvecs[i][right_id], tvecs[i][right_id]);
        common_view_with_maxPts.push_back(cv::Point2i(left_id, right_id));
        cout<<i<<" th cam: left/ right max id: "<<left_id<<", "<<right_id<<endl;
        cout<<i<<" th cam relative pose: "<<endl;
        cout<<rel_pose_btw2[i]<<endl;
    }
    
    camPoses_vec[0] = (cv::Mat_<double>(6, 1)<< 0, 0, 0, 0, 0, 0);
    camPoses_mat[0] = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat prev = cv::Mat::eye(4, 4, CV_64F);
    for(int ii = 1; ii < numCams; ii++)
    {
        cv::Mat newRef_frame_pose = rel_pose_btw2[ii]*prev;
        cout<<ii<<"the newRef_frame_pose: "<<endl;
        cout<<newRef_frame_pose<<endl;
        cv::Mat tvec = newRef_frame_pose(Range(0,3), Range(3,4));
        cv::Mat rmat = newRef_frame_pose(Range(0,3), Range(0,3));
        cv::Mat rvec;
        cv::Rodrigues(rmat, rvec);
        newRef_frame_pose.copyTo(camPoses_mat[ii]);
        camPoses_vec[ii] = (cv::Mat_<double>(6, 1)<< rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                                                     tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        newRef_frame_pose.copyTo(prev); // check wheather the value is changing
    }
    cout<<endl;
    cout<<"//-------- Before Optimization --------//"<<endl;
    for(int i=0; i<numCams; i++){
        cout<<i<<" th cam"<<endl;
        calibQuality_before(Ks[i], calib.objPts, mv_charucoCorners[i], mv_charuoIdx[i], vis_2view[i], Rvecs[i], tvecs[i], 5.0, gray_pattern_images[i]);
    }

    for(int i=0; i<numCams; i++)
        fill_empty_space(calib.objPts.size(), vis_2view[i], mv_charuoIdx[i], mv_charucoCorners[i], filled_mv_charucoCorners[i]);

    // multicam calibration using sparse bundle adjustment
    calib.multiCamCalibration(Ks, calib.objPts, filled_mv_charucoCorners, mv_charuoIdx, Rvecs, tvecs, camPoses_mat, camPoses_vec, patPoses_vec);

    // using corresponding points of model image, find location of the 3d face mesh
    vector<cv::Point2f> corrPts_l, corrPts_r;
    load_user_input(corrPts_l, corrPts_r);
    cv::Mat relpose = estimateRT_btw_mesh_n_refCam(Ks[2], Ks[3], camPoses_vec[2], camPoses_vec[3], corrPts_l, corrPts_r);
    save_calib_result("calib_output.txt", Ks, camPoses_vec, relpose);

    cout<<"========== after optimization =========="<<endl;
    for(int i = 0; i < numCams; i++)
    {
        cout<<i<<" th camera."<<endl;
        calibQuality_after(Ks[i], calib.objPts, mv_charucoCorners[i], mv_charuoIdx[i], camPoses_vec[i], patPoses_vec[0], vis_2view[0], vis_2view[i]);
    }

    cout<<endl;
    cout<<"after optimization!"<<endl;
    cout<<"## cam 0 and cam 1 ##"<<endl;
    check_relative_transf(Ks[0], Ks[1], calib.objPts, mv_charucoCorners[0][6], mv_charuoIdx[0][6], 
        mv_charucoCorners[1][6], mv_charuoIdx[1][6], camPoses_vec[1], patPoses_vec[0][6], patPoses_vec[1][6]);
    cout<<endl;
    cout<<"## cam 0 and cam 4 ##"<<endl;
    check_relative_transf(Ks[0], Ks[4], calib.objPts, mv_charucoCorners[0][6], mv_charuoIdx[0][6], 
        mv_charucoCorners[4][6], mv_charuoIdx[4][6], camPoses_vec[4], patPoses_vec[0][6], patPoses_vec[4][6]);

    cout<<endl;
    cout<<"tri_reproErr"<<endl;
    for(int i=1; i<numCams; i++)
    {
        cout<<i<<" th camera./ tri_reproE"<<endl;
        tri_reproErr(calib.objPts.size(), Ks[0], Ks[i], mv_charucoCorners[0][6], mv_charuoIdx[0][6], 
                     mv_charucoCorners[i][6], mv_charuoIdx[i][6], camPoses_vec[i]);
    }
    return 0;
}

void save_calib_result(const string& filename, const vector<sfmlib::Intrinsics>& Ks, const vector<cv::Mat>& camPose_vec, const cv::Mat& mesh2ref)
{
    ofstream outFile(filename);
    outFile<<"focal_x, focal_y, princ_x, princ_y, k1, k2(ki are radial distortion parameters, where i={1,2})"<<endl;
    outFile<<"rx, ry, rz, tx, ty, tz(location of each camera from the referece camera"<<endl;
    outFile<<endl;
    for(int i=0; i<Ks.size(); i++)
    {
        outFile<<i<<" th camera"<<endl;
        outFile<<Ks[i].focal_x<<" "<<Ks[i].focal_y<<" "<<Ks[i].princ_x<<" "<<Ks[i].princ_y<<" "<<Ks[i].k1<<" "<<Ks[i].k2<<" "<<endl;
        outFile<<camPose_vec[i].at<double>(0)<<" "<<camPose_vec[i].at<double>(1)<<" "<<camPose_vec[i].at<double>(2)<<" "<<
                 camPose_vec[i].at<double>(3)<<" "<<camPose_vec[i].at<double>(4)<<" "<<camPose_vec[i].at<double>(5)<<endl;
        cv::Mat rvec = (cv::Mat_<double>(3, 1)<< camPose_vec[i].at<double>(0), camPose_vec[i].at<double>(1), camPose_vec[i].at<double>(2));
        cv::Mat Rmat;
        cv::Rodrigues(rvec, Rmat);
        outFile<<"rotation matrix"<<endl;
        outFile<<Rmat.at<double>(0, 0)<<" "<<Rmat.at<double>(0, 1)<<" "<<Rmat.at<double>(0, 2)<<endl;
        outFile<<Rmat.at<double>(1, 0)<<" "<<Rmat.at<double>(1, 1)<<" "<<Rmat.at<double>(1, 2)<<endl;
        outFile<<Rmat.at<double>(2, 0)<<" "<<Rmat.at<double>(2, 1)<<" "<<Rmat.at<double>(2, 2)<<endl;
        outFile<<endl;
    }
    outFile<<"relative pose btw mesh and reference camera (mesh to refcam)."<<endl;
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            outFile<<mesh2ref.at<float>(i, j)<<" ";
        }
        outFile<<endl;
    }
        
    outFile.close();
}

Mat estimateRT_btw_mesh_n_refCam(const sfmlib::Intrinsics& lcam, const sfmlib::Intrinsics& rcam, cv::Mat& camPoses_l, cv::Mat& camPoses_r,
    const vector<cv::Point2f>& lpts, const vector<cv::Point2f>& rpts)
{
    camPoses_l.convertTo(camPoses_l, CV_32F);
    cv::Mat rvec = (cv::Mat_<float>(3, 1)<< camPoses_l.at<float>(0), camPoses_l.at<float>(1), camPoses_l.at<float>(2));
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    cv::Mat lRt(3, 4, CV_32F);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            lRt.at<float>(i, j) = rmat.at<float>(i, j);
        }
    }
    lRt.at<float>(0, 3) = camPoses_l.at<float>(3);
    lRt.at<float>(1, 3) = camPoses_l.at<float>(4);
    lRt.at<float>(2, 3) = camPoses_l.at<float>(5);
    cout<<"lRt"<<endl;
    cout<<lRt<<endl;

    camPoses_r.convertTo(camPoses_r, CV_32F);
    cv::Mat rvec2 = (cv::Mat_<float>(3, 1)<< camPoses_r.at<float>(0), camPoses_r.at<float>(1), camPoses_r.at<float>(2));
    cv::Mat rmat2;
    cv::Rodrigues(rvec, rmat2);
    cv::Mat rRt(3, 4, CV_32F);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            rRt.at<float>(i, j) = rmat2.at<float>(i, j);
        }
    }
    rRt.at<float>(0, 3) = camPoses_r.at<float>(3);
    rRt.at<float>(1, 3) = camPoses_r.at<float>(4);
    rRt.at<float>(2, 3) = camPoses_r.at<float>(5);
    cout<<"rRt"<<endl;
    cout<<rRt<<endl;

    cv::Mat lK = (cv::Mat_<float>(3, 3)<<
                    lcam.focal_x, 0, lcam.princ_x,
                    0, lcam.focal_y, lcam.princ_y,
                    0, 0, 1);
    cv::Mat rK = (cv::Mat_<float>(3, 3)<<
                    rcam.focal_x, 0, rcam.princ_x,
                    0, rcam.focal_y, rcam.princ_y,
                    0, 0, 1);
    cout<<"lK"<<endl;
    cout<<lK<<endl;
  
    cv::Mat dist_l = (cv::Mat_<float>(1, 5)<< lcam.k1, lcam.k2, 0, 0, 0);
    cv::Mat dist_r = (cv::Mat_<float>(1, 5)<< rcam.k1, rcam.k2, 0, 0, 0);

    cv::Mat lP = lK*lRt;
    cv::Mat rP = rK*rRt;

    cv::Mat X3d(4, lpts.size(), CV_32F);
    cv::triangulatePoints(lP, rP, lpts, rpts, X3d);
    vector<cv::Point3f> p3D;
    p3D.reserve(lpts.size());
    cout<<"---- after triangulation ----"<<endl;
    for(int i = 0; i < lpts.size(); i++)
    {
        cv::Point3f P;
        float z_norm = 1.0/X3d.at<float>(3, i);
        P.x = X3d.at<float>(0, i)*z_norm;
        P.y = X3d.at<float>(1, i)*z_norm;
        P.z = X3d.at<float>(2, i)*z_norm;
        p3D.push_back(P);
        cout<<P.x<<" " <<P.y<<" " <<P.z<<endl;
    }

    // solvePnP
    cv::Mat rvec_, tvec_, rmat_;
    cv::solvePnP(p3D, lpts, lK, dist_l, rvec_, tvec_);
    cv::Rodrigues(rvec_, rmat_);
    cv::Mat mesh_2_lcam = (cv::Mat_<float>(4, 4)<< 
        rmat_.at<float>(0, 0), rmat_.at<float>(0, 1), rmat_.at<float>(0, 2), tvec_.at<float>(0),
        rmat_.at<float>(1, 0), rmat_.at<float>(1, 1), rmat_.at<float>(1, 2), tvec_.at<float>(1),
        rmat_.at<float>(2, 0), rmat_.at<float>(2, 1), rmat_.at<float>(2, 2), tvec_.at<float>(2),
        0, 0, 0, 1);
    
    cv::Mat ref_2_lcam = cv::Mat::zeros(4, 4, CV_32F);
    for(int i=0; i<3; i++)
    for(int j=0; j<4; j++)
        ref_2_lcam.at<float>(i, j) = lRt.at<float>(i, j);
    ref_2_lcam.at<float>(3, 3) = 1.0;
    cv::Mat relpose_btw_mesh_and_refCam = ref_2_lcam.inv()*mesh_2_lcam;
    return relpose_btw_mesh_and_refCam;
}

//maybe this is wrong.. in previous sample data, we used all images but now it is different
void fill_empty_space(const int& num_whole_corners, const visibility_per_img& vis, const idx_per_cam& mv_charuoIdx, 
                      const corners_per_cam& mv_charucoCorners, corners_per_cam& filled_mv_charucoCorners)
{
    int num_whole_imgs = vis.size();
    cout<<"num whole imgs in fill_empty_space: "<<num_whole_imgs<<endl;
    filled_mv_charucoCorners.resize(num_whole_imgs);
    int charIdx_from_whole_data = 0;

    for(int ii = 0; ii < num_whole_imgs; ii++)
    {
        if(vis[ii] > 0)
        { // if camera captured any corners
            
            filled_mv_charucoCorners[ii].resize(num_whole_corners);

            int k = 0;
            for(int j = 0; j < num_whole_corners; j++)
            {
                if(mv_charuoIdx[charIdx_from_whole_data][k] == j){
                    filled_mv_charucoCorners[ii][j] = mv_charucoCorners[charIdx_from_whole_data][k];
                    k++;
                }
                else{
                    filled_mv_charucoCorners[ii][j] = cv::Point2f(0, 0);
                }
            }
            charIdx_from_whole_data++;
        }
        else filled_mv_charucoCorners[ii].clear(); // to skip those image frames  
    }
    cout<<"finished filling empty space"<<endl;
}


void calibQuality_before(const sfmlib::Intrinsics& intrins,
            const vector<cv::Point3f>& world3d,
                  corners_per_cam& mv_charucoCorners,
                  idx_per_cam& multi_view_charuoIdx,
            const visibility_per_img& vis,
            const vector<cv::Mat>& Rvec_pattern2cam,
            const vector<cv::Mat>& Tvec_pattern2cam,
            const float& reprojErr_thres,
            const vector<cv::Mat>& pattern_imgs)
{
    int nImg = multi_view_charuoIdx.size();
    int numWholePts = 0;
    double reproErr_per_cam = 0;
    double max = 0;
    int max_err_imgIdx, max_err_ptIdx;
    int next_id=0;
    int img_index;
    cout<<"#imgs in calibQual_before: "<<nImg<<endl;
    for(int j=0; j<nImg; j++)
    {
        for(int h=next_id; h<vis.size(); h++)
        {
            if(vis[h]>0){next_id = h + 1; img_index=h; break;}
        }
        cv::Mat rmat;
        cv::Rodrigues(Rvec_pattern2cam[j], rmat);
        cv::Mat Rt = (cv::Mat_<double>(4, 4)<< 
            rmat.at<double>(0, 0), rmat.at<double>(0, 1), rmat.at<double>(0, 2), Tvec_pattern2cam[j].at<double>(0),
            rmat.at<double>(1, 0), rmat.at<double>(1, 1), rmat.at<double>(1, 2), Tvec_pattern2cam[j].at<double>(1),
            rmat.at<double>(2, 0), rmat.at<double>(2, 1), rmat.at<double>(2, 2), Tvec_pattern2cam[j].at<double>(2),
            0, 0, 0, 1);

        cv::Mat copied_img;
        pattern_imgs[img_index].copyTo(copied_img);
        numWholePts += multi_view_charuoIdx[j].size();
        for(int k=0; k<multi_view_charuoIdx[j].size(); k++)
        {
            int id = multi_view_charuoIdx[j][k];
            cv::Mat PP = (cv::Mat_<double>(4, 1)<< world3d[id].x, world3d[id].y, world3d[id].z, 1);
            cv::Point2f pp = cv::Point2f(mv_charucoCorners[j][k]);
            cv::Mat pat_PP = Rt*PP;
            cv::Point2f up;
            up.x = pat_PP.at<double>(0)/pat_PP.at<double>(2);
            up.y = pat_PP.at<double>(1)/pat_PP.at<double>(2);
            float r2 = up.x*up.x + up.y*up.y;
            float dist = 1.0 + (intrins.k1+ intrins.k2*r2)*r2;
            cv::Point2f dp;
            dp.x = intrins.focal_x*up.x*dist+intrins.princ_x;
            dp.y = intrins.focal_y*up.y*dist+intrins.princ_y;

            double sqr_err = (dp.x-pp.x)*(dp.x-pp.x) + (dp.y-pp.y)*(dp.y-pp.y);
            cv::circle(copied_img, dp, 6, cv::Scalar(200, 0, 0));
            if(sqrt(sqr_err) > reprojErr_thres)
            {
                cv::circle(copied_img, mv_charucoCorners[j][k], 6, cv::Scalar(200, 0, 0));
                max_err_imgIdx = j;
                max_err_ptIdx = id;
                cout<<"distance error: "<<sqrt(sqr_err)<<" remove the point "<<max_err_ptIdx<<" in"<<max_err_imgIdx<<" th image"<<endl;
                multi_view_charuoIdx[j].erase(multi_view_charuoIdx[j].begin()+ k);
                mv_charucoCorners[j].erase(mv_charucoCorners[j].begin()+ k);
                if(k < multi_view_charuoIdx[j].size()-1) k--;
                if(max<sqr_err) 
                {
                    max = sqrt(sqr_err);
                }
            }
            reproErr_per_cam += sqrt(sqr_err);
        }
        cv::imwrite(cv::format("before/detected_aruco_corners%02d.png", j), copied_img);
    }
    reproErr_per_cam=(float)reproErr_per_cam/numWholePts;
    cout<<"repro:  ---"<<reproErr_per_cam<<endl;
    cout<<"completed 'calibQual_before'"<<endl;
}

void calibQuality_after(const sfmlib::Intrinsics& intrins,
            const vector<cv::Point3f>& world3d,
            const corners_per_cam& mv_corners,
            const idx_per_cam& mv_idx,
            const cv::Mat& sample_camPoses,
            const vector<cv::Mat>& ref_patPoses,
            const visibility_per_img& vis_ref,
            const visibility_per_img& vis)
{
    {
        int nWholeImg = vis_ref.size();
        int numWholePts = 0;
        double max_error =0;
        double reproErr_per_cam = 0;
        cv::Mat camPoses_rvec = (cv::Mat_<double>(3, 1)<< sample_camPoses.at<double>(0), sample_camPoses.at<double>(1), sample_camPoses.at<double>(2));
        cv::Mat camPoses_rmat;
        cv::Rodrigues(camPoses_rvec, camPoses_rmat);
        cv::Mat camPoseMat = (cv::Mat_<double>(4, 4)<<
            camPoses_rmat.at<double>(0, 0), camPoses_rmat.at<double>(0, 1), camPoses_rmat.at<double>(0, 2), sample_camPoses.at<double>(3),
            camPoses_rmat.at<double>(1, 0), camPoses_rmat.at<double>(1, 1), camPoses_rmat.at<double>(1, 2), sample_camPoses.at<double>(4),
            camPoses_rmat.at<double>(2, 0), camPoses_rmat.at<double>(2, 1), camPoses_rmat.at<double>(2, 2), sample_camPoses.at<double>(5),
            0, 0, 0, 1);

        int max_err_imgIdx, max_err_ptIdx;
        int ref_seen_view_idx = -1;
        int seen_view_idx = -1;
        for(int j = 0; j < nWholeImg; j++)
        {
            if(vis_ref[j] > 1) ref_seen_view_idx++;
            if(vis[j] > 1) seen_view_idx++;
            else continue;
            if(vis_ref[j] < 1) continue;

            int ref_id = ref_seen_view_idx;
            int nhd_id = seen_view_idx;
            cv::Mat patRmat;
            cv::Mat patRvec = (cv::Mat_<double>(3, 1)<< ref_patPoses[ref_id].at<double>(0), ref_patPoses[ref_id].at<double>(1), ref_patPoses[ref_id].at<double>(2));
            //cout<<"patRvec: "<<patRvec<<endl;
            cv::Rodrigues(patRvec, patRmat);
            cv::Mat ref_patPose = (cv::Mat_<double>(4, 4)<<
                patRmat.at<double>(0, 0), patRmat.at<double>(0, 1), patRmat.at<double>(0, 2), ref_patPoses[ref_id].at<double>(3),
                patRmat.at<double>(1, 0), patRmat.at<double>(1, 1), patRmat.at<double>(1, 2), ref_patPoses[ref_id].at<double>(4),
                patRmat.at<double>(2, 0), patRmat.at<double>(2, 1), patRmat.at<double>(2, 2), ref_patPoses[ref_id].at<double>(5),
                0, 0, 0, 1);

            int nPts = mv_idx[nhd_id].size();
            numWholePts += nPts;
        
            for(int k=0; k<nPts; k++)
            {
                int id = mv_idx[nhd_id][k];
                cv::Mat PP = (cv::Mat_<double>(4, 1)<< world3d[id].x, world3d[id].y, world3d[id].z, 1);
                cv::Mat ref_p2d = ref_patPose*PP;
                cv::Mat cam_inv_patPP = camPoseMat*ref_p2d;
                cv::Point2f up;
                up.x = cam_inv_patPP.at<double>(0)/cam_inv_patPP.at<double>(2);
                up.y = cam_inv_patPP.at<double>(1)/cam_inv_patPP.at<double>(2);
                float r2 = up.x*up.x + up.y*up.y;
                float dist = 1.0 + (intrins.k1+ intrins.k2*r2)*r2;
                cv::Point2f dp;
                dp.x = intrins.focal_x*up.x*dist+intrins.princ_x;
                dp.y = intrins.focal_y*up.y*dist+intrins.princ_y;
                cv::Point2f pp = cv::Point2f(mv_corners[nhd_id][k]);

                double sqr_err =(dp.x-pp.x)*(dp.x-pp.x) + (dp.y-pp.y)*(dp.y-pp.y);
                if(max_error< sqr_err) {
                    max_error = sqr_err;
                    max_err_imgIdx = j;
                    max_err_ptIdx = id;
                    cout<<"errouneos pixels: "<<pp.x<<" "<<dp.x<<" "<<pp.y<<" "<<dp.y<<endl;
                }
                reproErr_per_cam += sqrt(sqr_err);
            }
        }
        reproErr_per_cam=(float)reproErr_per_cam/numWholePts;
        cout<<"max error:  "<<sqrt(max_error)<<endl;
        cout<<"repro:  ---"<<reproErr_per_cam<<endl;
        cout<<max_err_imgIdx<<" th image, "<<max_err_ptIdx<<" index point"<<endl;
    }
}

void tri_reproErr(const int& nWholeCorners, const sfmlib::Intrinsics& lcam, const sfmlib::Intrinsics& rcam, 
    const corners_per_img& pl, const idx_per_img& id_l, const corners_per_img& pr, const idx_per_img& id_r, cv::Mat& camPoses_vec)
{
    cv::Mat lRt = cv::Mat::zeros(3, 4, CV_32F);
    lRt.at<float>(0, 0) =1;
    lRt.at<float>(1, 1) =1;
    lRt.at<float>(2, 2) =1;
    
    camPoses_vec.convertTo(camPoses_vec, CV_32F);
    cv::Mat rvec = (cv::Mat_<float>(3, 1)<< camPoses_vec.at<float>(0), camPoses_vec.at<float>(1), camPoses_vec.at<float>(2));
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    cv::Mat rRt(3, 4, CV_32F);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            rRt.at<float>(i, j) = rmat.at<float>(i, j);
        }
    }
    
    rRt.at<float>(0, 3) = camPoses_vec.at<float>(3);
    rRt.at<float>(1, 3) = camPoses_vec.at<float>(4);
    rRt.at<float>(2, 3) = camPoses_vec.at<float>(5);

    cv::Mat lK = (cv::Mat_<float>(3, 3)<<
                    lcam.focal_x, 0, lcam.princ_x,
                    0, lcam.focal_y, lcam.princ_y,
                    0, 0, 1);
    cv::Mat rK = (cv::Mat_<float>(3, 3)<<
                    rcam.focal_x, 0, rcam.princ_x,
                    0, rcam.focal_y, rcam.princ_y,
                    0, 0, 1);
    
    cv::Mat dist_l = (cv::Mat_<float>(1, 5)<< lcam.k1, lcam.k2, 0, 0, 0);
    cv::Mat dist_r = (cv::Mat_<float>(1, 5)<< rcam.k1, rcam.k2, 0, 0, 0);

    cv::Mat lP = lK*lRt;
    cv::Mat rP = rK*rRt;

    vector<cv::Point3f> p3D;
    vector<int> common_ptsIdx;
    std::set_intersection(id_l.begin(), id_l.end(),
                          id_r.begin(), id_r.end(),
                          std::back_inserter(common_ptsIdx));
    vector<cv::Point2f> common_pl, common_pr; 
    int cnt_l = 0;
    int cnt_r = 0;
    for(int i=0; i<common_ptsIdx.size(); i++)
    {
        while(common_ptsIdx[i] != id_l[cnt_l]) cnt_l++;
        while(common_ptsIdx[i] != id_r[cnt_r]) cnt_r++;
        common_pl.push_back(pl[cnt_l]);
        common_pr.push_back(pr[cnt_r]);
    }
   
    {
        int nPts = common_ptsIdx.size();
        cv::Mat X3d(4, nPts, CV_32F);//4xN array of reconstructed points in homogeneous coordinates.
        cv::triangulatePoints(lP, rP, common_pl, common_pr, X3d);
        p3D.clear();
        p3D.reserve(nPts);
        for(int i=0; i<nPts; i++)
        {
            cv::Point3f P;
            double z_norm = 1.0/X3d.at<float>(3, i);
            P.x = X3d.at<float>(0, i)*z_norm;
            P.y = X3d.at<float>(1, i)*z_norm;
            P.z = X3d.at<float>(2, i)*z_norm;
            p3D.push_back(P);
        }
        cout<<"---- after triangulation ----"<<endl;

        // reproject points
        vector<cv::Point2f> projectedPts;
        rvec= (cv::Mat_<float>(3, 1) << 0, 0, 0);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);
        cv::projectPoints(p3D, rvec, tvec, lK, dist_l, projectedPts);
        double err = 0;
        for(int i=0; i<id_l.size(); i++){
            double err_x = pl[i].x - projectedPts[i].x;
            double err_y = pl[i].y - projectedPts[i].y;
            err += sqrt(err_x*err_x + err_y*err_y);
            cout<<"projected pts_x: "<<pl[i].x<<", "<<projectedPts[i].x<<"/ y: "<<pl[i].y<<", "<<projectedPts[i].y<<endl;
        }
        cout<<"reprojection error: "<< (double)err/id_l.size()<<endl;
    }
    
}



void check_relative_transf(const sfmlib::Intrinsics& Kl, const sfmlib::Intrinsics& Kr, const vector<cv::Point3f>& world3d, 
                           const corners_per_img& pl, const idx_per_img& id_l, const corners_per_img& pr, 
                           const idx_per_img& id_r, const cv::Mat& camPoses_vec, cv::Mat& patPose_l, cv::Mat& patPose_r)
{
    cout<< "---- check_relative_transf ----"<<endl;
    // lRt: 4x4 matrix
    if(id_l.size() == id_r.size()){
        int nPts = id_l.size();
        cv::Mat lK = (cv::Mat_<float>(3, 3)<<
            Kl.focal_x, 0, Kl.princ_x,
            0, Kl.focal_y, Kl.princ_y, 
            0, 0, 1);
        cv::Mat rK = (cv::Mat_<float>(3, 3)<<
            Kr.focal_x, 0, Kr.princ_x,
            0, Kr.focal_y, Kr.princ_y, 
            0, 0, 1);
        cv::Mat dist_l = (cv::Mat_<float>(1, 5)<< Kl.k1, Kl.k2, 0, 0, 0);
        cv::Mat dist_r = (cv::Mat_<float>(1, 5)<< Kr.k1, Kr.k2, 0, 0, 0);

        vector<cv::Point3f> P;
        for(int i=0; i<nPts; i++)
        {
           P.push_back(world3d[id_l[i]]);
        }
        #if 0
        cv::Mat rvec, tvec;
        cv::solvePnP(P, pl, lK, dist_l, rvec, tvec);
        #else
        patPose_l.convertTo(patPose_l, CV_32F);
        cv::Mat rvec = (cv::Mat_<float>(3, 1)<< patPose_l.at<float>(0), patPose_l.at<float>(1), patPose_l.at<float>(2));
        cv::Mat tvec = (cv::Mat_<float>(3, 1)<< patPose_l.at<float>(3), patPose_l.at<float>(4), patPose_l.at<float>(5));

        patPose_r.convertTo(patPose_r, CV_32F);
        cv::Mat rrvec = (cv::Mat_<float>(3, 1)<< patPose_r.at<float>(0), patPose_r.at<float>(1), patPose_r.at<float>(2));
        cv::Mat rtvec = (cv::Mat_<float>(3, 1)<< patPose_r.at<float>(3), patPose_r.at<float>(4), patPose_r.at<float>(5));
        #endif
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        patPose_l = (cv::Mat_<float>(4, 4)<< 
            R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), tvec.at<float>(0),
            R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), tvec.at<float>(1),
            R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), tvec.at<float>(2),
            0, 0, 0, 1);
        
        cv::Rodrigues(rrvec, R);
        patPose_r = (cv::Mat_<float>(4, 4)<< 
            R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), rtvec.at<float>(0),
            R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), rtvec.at<float>(1),
            R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), rtvec.at<float>(2),
            0, 0, 0, 1);

        cout<<"camPoses_vec: "<<camPoses_vec<<endl;
        cv::Mat camPoses_vec_;
        camPoses_vec.copyTo(camPoses_vec_);
        camPoses_vec_.convertTo(camPoses_vec_, CV_32F);
        cout<<"after type conversion: "<<camPoses_vec_<<endl;
        cv::Mat camPoses_rvec = (cv::Mat_<float>(3, 1)<< camPoses_vec_.at<float>(0), camPoses_vec_.at<float>(1), camPoses_vec_.at<float>(2));
        cv::Mat camPoses_rmat;
        cv::Rodrigues(camPoses_rvec, camPoses_rmat);
        cv::Mat camPoseMat = (cv::Mat_<float>(4, 4)<<
            camPoses_rmat.at<float>(0, 0), camPoses_rmat.at<float>(0, 1), camPoses_rmat.at<float>(0, 2), camPoses_vec_.at<float>(3),
            camPoses_rmat.at<float>(1, 0), camPoses_rmat.at<float>(1, 1), camPoses_rmat.at<float>(1, 2), camPoses_vec_.at<float>(4),
            camPoses_rmat.at<float>(2, 0), camPoses_rmat.at<float>(2, 1), camPoses_rmat.at<float>(2, 2), camPoses_vec_.at<float>(5),
            0, 0, 0, 1);

        
        cout<<"--- left view ---"<<endl;
        vector<cv::Point2f> est_pl;
        for(int i=0; i<nPts; i++)
        {
            cv::Mat p = (cv::Mat_<float>(4, 1)<< P[i].x, P[i].y, P[i].z, 1);
            cv::Mat Rt1X3d = patPose_l*p;

            cv::Point2f up;
            up.x = Rt1X3d.at<float>(0)/Rt1X3d.at<float>(2);
            up.y = Rt1X3d.at<float>(1)/Rt1X3d.at<float>(2);
            float r2 = up.x*up.x + up.y*up.y;
            float dist = 1.0 + (Kl.k1+ Kl.k2*r2)*r2;
            cv::Point2f dp;
            dp.x = Kl.focal_x*up.x*dist+Kl.princ_x;
            dp.y = Kl.focal_y*up.y*dist+Kl.princ_y;
            est_pl.push_back(dp);
        }
        float reproj_err = 0;
        for(int i=0; i<nPts; i++)
        {
            float err_x = pl[i].x - est_pl[i].x;
            float err_y = pl[i].y - est_pl[i].y;
            reproj_err +=sqrt(err_x*err_x + err_y*err_y);
            cout<<i<<"th diff: "<<pl[i].x<<", "<<est_pl[i].x<<" / "<<pl[i].y<<", "<<est_pl[i].y<<endl;
        }
        reproj_err=(float)reproj_err/nPts;
        cout<<"/--- final reprojection error: "<<reproj_err<<" ---/"<<endl;
        cout<<endl;
        cout<<endl;

        cout<<"--- right view ---"<<endl;
        vector<cv::Point2f> est_pr;
        for(int i=0; i<nPts; i++)
        {
            cv::Mat p = (cv::Mat_<float>(4, 1)<< P[i].x, P[i].y, P[i].z, 1);
            cv::Mat Rt1X3d = patPose_l*p;
            cv::Mat Rt2X3d = Rt1X3d;
            cv::Mat Rt12P = camPoseMat*Rt2X3d;

            cv::Point2f up;
            up.x = Rt12P.at<float>(0)/Rt12P.at<float>(2);
            up.y = Rt12P.at<float>(1)/Rt12P.at<float>(2);
            float r2 = up.x*up.x + up.y*up.y;
            float dist = 1.0 + (Kr.k1+ Kr.k2*r2)*r2;
            cv::Point2f dp;
            dp.x = Kr.focal_x*up.x*dist+Kr.princ_x;
            dp.y = Kr.focal_y*up.y*dist+Kr.princ_y;
            est_pr.push_back(dp);
        }
        reproj_err = 0;
        for(int i=0; i<nPts; i++)
        {
            float err_x = pr[i].x - est_pr[i].x;
            float err_y = pr[i].y - est_pr[i].y;
            reproj_err +=sqrt(err_x*err_x + err_y*err_y);
            cout<<i<<"th diff: "<<pr[i].x<<", "<<est_pr[i].x<<" / "<<pr[i].y<<", "<<est_pr[i].y<<endl;
        }
        reproj_err=(float)reproj_err/nPts;
        cout<<"/--- final reprojection error: "<<reproj_err<<" ---/"<<endl;
    }
}

void find_common_scene_index_for_twoView(const vector<int>& v1, const vector<int>& v2, cv::Point2i& idx)
{
    int max = -1000;
    int sz = v1.size();
    int id_v1 = -1;
    int id_v2 = -1;
    if(sz!=v2.size()) printf("Size is different! Error!\n");
    else{
        for(int i=0; i<sz; i++){
            if(v1[i]>0) id_v1++;
            if(v2[i]>0) id_v2++;
            if(v1[i]>0 && v2[i]>0){
                int numSeenPts_in_common = std::min(v1[i], v2[i]);
                if(max<numSeenPts_in_common){
                    max = numSeenPts_in_common;
                    idx.x = id_v1;
                    idx.y = id_v2;
                }
            }
        }
    }
}

cv::Mat computeRelativeRt(const cv::Mat& rvec1, const cv::Mat& t1,const cv::Mat& rvec2, const cv::Mat& t2)
{
    // relative transformation btw two camera as the left camera setting center
    cv::Mat R1, R2;
    cv::Rodrigues(rvec1, R1);
    cv::Rodrigues(rvec2, R2);
    cv::Mat Rt1 = (cv::Mat_<double>(4, 4)<< R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0),
                                            R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1),
                                            R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2),
                                            0, 0, 0, 1);
    cv::Mat Rt2 = (cv::Mat_<double>(4, 4)<< R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0),
                                            R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1),
                                            R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2),
                                            0, 0, 0, 1);
    cv::Mat Rt_from_1to2 = Rt2*Rt1.inv();
    return Rt_from_1to2;
}


cv::Mat downsize_img(const cv::Mat& img, const float downSizeScale)
{
    // downsize image to quarter size
    cv::Mat downSizedImg;
    cv::resize(img, downSizedImg, cv::Size(0, 0), downSizeScale, downSizeScale, cv::INTER_AREA);
    return downSizedImg;
}

string extractIntegerWords(const string str){
    stringstream ss;
    ss << str;
    string temp;
    int found;
    while(!ss.eof()){
        ss>>temp;
        if(stringstream(temp)>>found)
            cout<<found<<"- ";
        temp ="";
    }
    return temp;
}

void loadImages(const string folderName, vector<cv::Mat>& imgs)
{
    imgs.clear();
    StringArray files;
    if( GetFileList(folderName, files, false)) 
    {
        printf("file size: %d\n", files.size());
    }
    
    int nImgs = files.size();
    files.sort();
    printf("Image loading... \n");

    for(int i=0; i<nImgs; i++){
        cv::Mat frame = cv::imread(folderName + "/" + files[i]);
        cv::Mat gray_image;
        cvtColor(frame, gray_image, COLOR_BGR2GRAY);
        if(frame.size().width>0){
            //cv::Mat downSz_img = downsize_img(gray_image, downSizeScale);
            imgs.push_back(gray_image);
        }
        else{
            printf("Image loading failed!\n");
            break;
        } 
    }
    printf("%d number of images are loaded.\n", imgs.size());
}

int readImagesFromVideo(const string filename, const string outFolderName, const int camIdx, const float downSizeScale, vector<cv::Mat>& imgs, int& n)
{
    VideoCapture sequence(filename);
    if(!sequence.isOpened()) {cout<<"Failed to open image sequence!"<<endl; return -1;}
    else {cout<<"Open File to save images";}
    int num_image = sequence.get(CAP_PROP_FRAME_COUNT);
    cout<<" /"<<num_image<<" images"<<endl;
    
    imgs.clear();
    int skip_frames = 30;
    int idx = 0;

    omp_set_num_threads(8);
    #pragma opm parallel for
    for(int i=0; i<num_image; i++){
        sequence.grab();
        if(i%skip_frames == 0)
        {
            cv::Mat frame;
            sequence.read(frame);
            if(frame.size().width>0){
                cv::Mat downSizedImg = downsize_img(frame, downSizeScale);
                imgs.push_back(downSizedImg);
            }
            else break;
        }
    }

    omp_set_num_threads(8);
    #pragma opm parallel for
    for(int i=0; i<imgs.size(); i++)
    {
        imwrite(outFolderName.c_str()+cv::format("%d-%d.png",camIdx, i), imgs[i]);
    }
    
    n=imgs.size();

    return 1;
}



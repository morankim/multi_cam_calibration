#include "BundleAdjustment.h"

namespace sfmlib
{

namespace BundleAdjustment
{
int nCameras;
}
using namespace BundleAdjustment;

template <typename T>
void print2d_array(const int& rows, const int& cols, T** P)
{
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
        {
            cout<< P[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

template <typename T>
void print1d_array(const int& rows, T* P)
{
    for(int i=0; i<rows; i++){
        cout<< P[i]<<" ";
    }
        cout<<endl;
}

template <typename T>
void invert_rotation_matrix(T Rmat[3][3])
{
    T rot[3][3];
    // copy Rmat
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            rot[i][j] = Rmat[i][j];

    // inverse of a rotation matrix -> transpose of a matrix
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Rmat[i][j] = rot[j][i];
}

template <typename T>
void copy_matrix_4x4(T** Rfrom, T** Rto)
{
    int row4 = 4;
    for(int i=0; i<row4; i++)
      for(int j=0; j<row4; j++){
          Rto[i][j] = Rfrom[i][j];
      }
}

template <typename T>
void eulerAnglesToRotationMatrix(const T *const extr, T rota_mat[3][3])
{
    T R_x[3][3];
    T R_y[3][3];
    T R_z[3][3];

    T theta = sqrt(extr[0]*extr[0]+extr[1]*extr[1]+extr[2]*extr[2]);

    if(theta > T(0.0000001)){
        R_x[0][0] = T(cos(theta));
        R_x[0][1] = T(0.0);
        R_x[0][2] = T(0.0);
        R_x[1][0] = T(0.0);
        R_x[1][1] = T(cos(theta));
        R_x[1][2] = T(0.0);
        R_x[2][0] = T(0.0);
        R_x[2][1] = T(0.0);
        R_x[2][2] = T(cos(theta));

        T rx =extr[0]/theta;
        T ry =extr[1]/theta;
        T rz =extr[2]/theta;

        T k = 1.0-cos(theta);
        R_y[0][0] = k*rx*rx;
        R_y[0][1] = k*rx*ry;
        R_y[0][2] = k*rx*rz;
        R_y[1][0] = k*rx*ry;
        R_y[1][1] = k*ry*ry;
        R_y[1][2] = k*ry*rz;
        R_y[2][0] = k*rx*rz;
        R_y[2][1] = k*ry*rz;
        R_y[2][2] = k*rz*rz;

        T h = sin(theta);

        R_z[0][0] = T(0.0);
        R_z[0][1] = -rz*h;
        R_z[0][2] =  ry*h;
        R_z[1][0] =  rz*h;
        R_z[1][1] = T(0.0);
        R_z[1][2] = -rx*h;
        R_z[2][0] = -ry*h;
        R_z[2][1] =  rx*h;
        R_z[2][2] = T(0.0);

        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            rota_mat[i][j] = R_x[i][j]+R_y[i][j]+R_z[i][j];
    }
    else
    {
        rota_mat[0][0] = T(1.0);
        rota_mat[0][1] = T(0.0);
        rota_mat[0][2] = T(0.0);
        rota_mat[1][0] = T(0.0);
        rota_mat[1][1] = T(1.0);
        rota_mat[1][2] = T(0.0);
        rota_mat[2][0] = T(0.0);
        rota_mat[2][1] = T(0.0);
        rota_mat[2][2] = T(1.0);
    }
}


template <typename T>
void invert_vector_4x4(const T *const T1, T R1_inv_4x4[4][4])
{
    T R1_inv[3][3];
    T t1[3];
    for(int i=0; i<3; i++)
        t1[i] = T1[3+i];
    
    eulerAnglesToRotationMatrix(T1, R1_inv);
    invert_rotation_matrix(R1_inv);
    for(int i=0; i<4; i++)
    for(int j=0; j<4; j++)
        R1_inv_4x4[i][j] = (T)0;

    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
        R1_inv_4x4[i][j] = R1_inv[i][j];
    R1_inv_4x4[0][3] = -(R1_inv[0][0]*t1[0]+R1_inv[0][1]*t1[1]+R1_inv[0][2]*t1[2]);
    R1_inv_4x4[1][3] = -(R1_inv[1][0]*t1[0]+R1_inv[1][1]*t1[1]+R1_inv[1][2]*t1[2]);
    R1_inv_4x4[2][3] = -(R1_inv[2][0]*t1[0]+R1_inv[2][1]*t1[1]+R1_inv[2][2]*t1[2]);
    R1_inv_4x4[3][3] = (T)1;
}

template <typename T>
void invert_matrix_4x4(T Rfrom[4][4], T Rto[4][4])
{
    T tvec[3];
    for(int i=0; i <3; i++)
        tvec[i] = Rfrom[i][3];

    for(int i=0; i<4; i++)
      for(int j=0; j<4; j++){
          Rto[j][i] = (T) 0;
      }

    for(int i=0; i<3; i++)
      for(int j=0; j<3; j++){
          Rto[j][i] = Rfrom[i][j];
      }
      Rto[0][3] = -Rfrom[0][0]*tvec[0] - Rfrom[1][0]*tvec[1] - Rfrom[2][0]*tvec[2];
      Rto[1][3] = -Rfrom[0][1]*tvec[0] - Rfrom[1][1]*tvec[1] - Rfrom[2][1]*tvec[2];
      Rto[2][3] = -Rfrom[0][2]*tvec[0] - Rfrom[1][2]*tvec[1] - Rfrom[2][2]*tvec[2];
      Rto[3][3] = (T) 1.0;
}

template <typename T>
void compute_relativeRt(const T *const T1, const T*const T2, T relRt_from_1to2[4][4])
{
    T R1_inv[3][3];
    T R1_inv_4x4[4][4];
    T R2[3][3];
    T t1[3], t2[3];
    for(int i=0; i<3; i++)
    {
        t1[i] = T1[3+i];
        t2[i] = T2[3+i];
    }
    //T* t2 = &T2[3];
    
    invert_vector_4x4(T1, R1_inv_4x4);
    eulerAnglesToRotationMatrix(T2, R2);

    relRt_from_1to2[0][0] = R2[0][0]*R1_inv_4x4[0][0] + R2[0][1]*R1_inv_4x4[1][0] + R2[0][2]*R1_inv_4x4[2][0] + t2[0]*R1_inv_4x4[3][0];
    relRt_from_1to2[0][1] = R2[0][0]*R1_inv_4x4[0][1] + R2[0][1]*R1_inv_4x4[1][1] + R2[0][2]*R1_inv_4x4[2][1] + t2[0]*R1_inv_4x4[3][1];
    relRt_from_1to2[0][2] = R2[0][0]*R1_inv_4x4[0][2] + R2[0][1]*R1_inv_4x4[1][2] + R2[0][2]*R1_inv_4x4[2][2] + t2[0]*R1_inv_4x4[3][2];
    relRt_from_1to2[0][3] = R2[0][0]*R1_inv_4x4[0][3] + R2[0][1]*R1_inv_4x4[1][3] + R2[0][2]*R1_inv_4x4[2][3] + t2[0]*R1_inv_4x4[3][3]; 
 
    relRt_from_1to2[1][0] = R2[1][0]*R1_inv_4x4[0][0] + R2[1][1]*R1_inv_4x4[1][0] + R2[1][2]*R1_inv_4x4[2][0] + t2[1]*R1_inv_4x4[3][0]; 
    relRt_from_1to2[1][1] = R2[1][0]*R1_inv_4x4[0][1] + R2[1][1]*R1_inv_4x4[1][1] + R2[1][2]*R1_inv_4x4[2][1] + t2[1]*R1_inv_4x4[3][1];
    relRt_from_1to2[1][2] = R2[1][0]*R1_inv_4x4[0][2] + R2[1][1]*R1_inv_4x4[1][2] + R2[1][2]*R1_inv_4x4[2][2] + t2[1]*R1_inv_4x4[3][2];
    relRt_from_1to2[1][3] = R2[1][0]*R1_inv_4x4[0][3] + R2[1][1]*R1_inv_4x4[1][3] + R2[1][2]*R1_inv_4x4[2][3] + t2[1]*R1_inv_4x4[3][3];

    relRt_from_1to2[2][0] = R2[2][0]*R1_inv_4x4[0][0] + R2[2][1]*R1_inv_4x4[1][0] + R2[2][2]*R1_inv_4x4[2][0] + t2[2]*R1_inv_4x4[3][0]; 
    relRt_from_1to2[2][1] = R2[2][0]*R1_inv_4x4[0][1] + R2[2][1]*R1_inv_4x4[1][1] + R2[2][2]*R1_inv_4x4[2][1] + t2[2]*R1_inv_4x4[3][1];
    relRt_from_1to2[2][2] = R2[2][0]*R1_inv_4x4[0][2] + R2[2][1]*R1_inv_4x4[1][2] + R2[2][2]*R1_inv_4x4[2][2] + t2[2]*R1_inv_4x4[3][2];
    relRt_from_1to2[2][3] = R2[2][0]*R1_inv_4x4[0][3] + R2[2][1]*R1_inv_4x4[1][3] + R2[2][2]*R1_inv_4x4[2][3] + t2[2]*R1_inv_4x4[3][3];

    relRt_from_1to2[3][0] = (T) 0.0;
    relRt_from_1to2[3][1] = (T) 0.0;
    relRt_from_1to2[3][2] = (T) 0.0;
    relRt_from_1to2[3][3] = (T) 1.0;
}

template <typename T>
void compute_relativeRt_ref(const T*const T2, T relRt_from_1to2[4][4])
{
    T R2[3][3];

    eulerAnglesToRotationMatrix(T2, R2);

    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    {
        relRt_from_1to2[i][j] = R2[i][j];
    }
    relRt_from_1to2[0][3] = T2[3];
    relRt_from_1to2[1][3] = T2[4];
    relRt_from_1to2[2][3] = T2[5];

    relRt_from_1to2[3][0] = (T) 0.0;
    relRt_from_1to2[3][1] = (T) 0.0;
    relRt_from_1to2[3][2] = (T) 0.0;
    relRt_from_1to2[3][3] = (T) 1.0;
}

struct ReprojectionError
{
    const float x1, y1, x2, y2;
    const float X, Y, Z;
    const float pcx, pcy;
    const int referenceFrameFlag;
    int numAverageResid;

    ReprojectionError(float p1x, float p1y, float p2x, float p2y, float Px, float Py, float Pz, float px, float py, int refFrame, int numResid) : 
        x1(p1x), y1(p1y), x2(p2x), y2(p2y), X(Px), Y(Py), Z(Pz), pcx(px), pcy(py), referenceFrameFlag(refFrame), numAverageResid(numResid) {}

    template <typename T>
    bool operator()(const T *const Intr1,
                    const T *const Intr2,
                    const T *const patTocam1,
                    const T *const patTocam2,
                    const T *const camPose1,
                    const T *const camPose2, 
                    T *residuals) const
    {

        // setting intrinsic parameters
        T fx_1 = Intr1[0];
        T fy_1 = Intr1[1];
        T cx_1 = (T) pcx;
        T cy_1 = (T) pcy;
        T k1_1 = Intr1[2];
        T k2_1 = Intr1[3];

        T fx_2 = Intr2[0];
        T fy_2 = Intr2[1];
        T cx_2 = (T) pcx;
        T cy_2 = (T) pcy;
        T k1_2 = Intr2[2];
        T k2_2 = Intr2[3];

        T rot[3][3];
        T relRt_1to2[4][4];
        T relRt_2to1[4][4];

        //Extrinsic[0,1,2]: rotation
        //Extrinsic[3,4,5]: translation
        //--------------  direction "1 -> 2"  --------------//
        eulerAnglesToRotationMatrix(patTocam1, rot); // convert X3d to cam1's coordinate points
        T X_1 = (T)X;
        T Y_1 = (T)Y;
        T Z_1 = (T)Z;

        T Rt_X1 = X_1 * rot[0][0] + Y_1 * rot[0][1] + Z_1 * rot[0][2] + patTocam1[3];
        T Rt_Y1 = X_1 * rot[1][0] + Y_1 * rot[1][1] + Z_1 * rot[1][2] + patTocam1[4];
        T Rt_Z1 = X_1 * rot[2][0] + Y_1 * rot[2][1] + Z_1 * rot[2][2] + patTocam1[5];
        T Rt_Z1_inv = (T)1.0/Rt_Z1;

        T xp1_ = Rt_X1 * Rt_Z1_inv;
        T yp1_ = Rt_Y1 * Rt_Z1_inv;
        T r2 = xp1_ * xp1_ + yp1_ * yp1_;
        T Kd = T(1.0) + (k1_1 + k2_1 * r2) * r2;
        // distorted 2d image point
        T direct_proj_x1 = fx_1 * (xp1_ * Kd) + cx_1;
        T direct_proj_y1 = fy_1 * (yp1_ * Kd) + cy_1;

        // setting for opposite direction of relative Rt for bidirectional computation
        if(referenceFrameFlag<0){
            compute_relativeRt(camPose1, camPose2, relRt_1to2);
        }
        else{
            compute_relativeRt_ref(camPose2, relRt_1to2);
        }

        T Rt1to2_transf_X1 =          Rt_X1 * relRt_1to2[0][0] + Rt_Y1 * relRt_1to2[0][1] + Rt_Z1 * relRt_1to2[0][2] + relRt_1to2[0][3];
        T Rt1to2_transf_Y1 =          Rt_X1 * relRt_1to2[1][0] + Rt_Y1 * relRt_1to2[1][1] + Rt_Z1 * relRt_1to2[1][2] + relRt_1to2[1][3];
        T Rt1to2_transf_Z1 = (T)1.0 /(Rt_X1 * relRt_1to2[2][0] + Rt_Y1 * relRt_1to2[2][1] + Rt_Z1 * relRt_1to2[2][2] + relRt_1to2[2][3]);

        //project X3d in cam2's coordinate to image plane
        T xp2 = Rt1to2_transf_X1 * Rt1to2_transf_Z1;
        T yp2 = Rt1to2_transf_Y1 * Rt1to2_transf_Z1;
          r2 = xp2 * xp2 + yp2 * yp2;
          Kd = T(1.0) + (k1_2 + k2_2 * r2) * r2;
        // distorted 2d image point
        T proj_x2 = fx_2 * (xp2 * Kd) + cx_2;
        T proj_y2 = fy_2 * (yp2 * Kd) + cy_2;
        //--------------------------------------------------//

        //--------------  direction "2 -> 1"  --------------//
        eulerAnglesToRotationMatrix(patTocam2, rot); // convert X3d to cam2's coordinate points
        T X_2 = (T)X;
        T Y_2 = (T)Y;
        T Z_2 = (T)Z;

        T Rt_X2 = X_2 * rot[0][0] + Y_2 * rot[0][1] + Z_2 * rot[0][2] + patTocam2[3];
        T Rt_Y2 = X_2 * rot[1][0] + Y_2 * rot[1][1] + Z_2 * rot[1][2] + patTocam2[4];
        T Rt_Z2 = X_2 * rot[2][0] + Y_2 * rot[2][1] + Z_2 * rot[2][2] + patTocam2[5];
        T Rt_Z2_inv = (T)1.0/Rt_Z2;

        T xp2_ = Rt_X2 * Rt_Z2_inv;
        T yp2_ = Rt_Y2 * Rt_Z2_inv;
          r2 = xp2_ * xp2_ + yp2_ * yp2_;
          Kd = T(1.0) + (k1_2 + k2_2 * r2) * r2;
        // distorted 2d image point
        T direct_proj_x2 = fx_2 * (xp2_ * Kd) + cx_2;
        T direct_proj_y2 = fy_2 * (yp2_ * Kd) + cy_2;

        // copy relRt_1to2 to relRt_2to1
        invert_matrix_4x4(relRt_1to2, relRt_2to1);
        
        //eulerAnglesToRotationMatrix(relRt_2to1, rot); // transform X3d in cam2's coord to cam1's coord.
        T Rt2to1_transf_X1 =          Rt_X2 * relRt_2to1[0][0] + Rt_Y2 * relRt_2to1[0][1] + Rt_Z2 * relRt_2to1[0][2] + relRt_2to1[0][3];
        T Rt2to1_transf_Y1 =          Rt_X2 * relRt_2to1[1][0] + Rt_Y2 * relRt_2to1[1][1] + Rt_Z2 * relRt_2to1[1][2] + relRt_2to1[1][3];
        T Rt2to1_transf_Z1 = (T)1.0 /(Rt_X2 * relRt_2to1[2][0] + Rt_Y2 * relRt_2to1[2][1] + Rt_Z2 * relRt_2to1[2][2] + relRt_2to1[2][3]);

        //project X3d in cam2's coordinate to image plane
        T xp1 = Rt2to1_transf_X1 * Rt2to1_transf_Z1;
        T yp1 = Rt2to1_transf_Y1 * Rt2to1_transf_Z1;
        //cout<<"xp1/ yp1: "<<xp1<<", "<<yp1<<endl;
    
          r2 = xp1 * xp1 + yp1 * yp1;
          Kd = T(1.0) + (k1_1 + k2_1 * r2) * r2;
        // distorted 2d image point
        T proj_x1 = fx_1 * (xp1 * Kd) + cx_1;
        T proj_y1 = fy_1 * (yp1 * Kd) + cy_1;

        residuals[0] = (T)x2 - proj_x2;
        residuals[1] = (T)y2 - proj_y2;
        residuals[2] = (T)x1 - proj_x1;
        residuals[3] = (T)y1 - proj_y1;
        residuals[4] = (T)x2 - direct_proj_x2;
        residuals[5] = (T)y2 - direct_proj_y2;
        residuals[6] = (T)x1 - direct_proj_x1;
        residuals[7] = (T)y1 - direct_proj_y1;
        
        return true;
    }

    static ceres::CostFunction *create(const float p1x, const float p1y, const float p2x, const float p2y, 
        const float Px, const float Py, const float Pz, const float px, const float py, const int refFrame, int numResid)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 8, 4, 4, 6, 6, 6, 6>(
            new ReprojectionError(p1x, p1y, p2x, p2y, Px, Py, Pz, px, py, refFrame, numResid)));
    }
};

void SparseBundleAdjustment::adjustBundle(vector<Intrinsics>& intrins,
                                          const vector<cv::Point3f>& world3d,
                                          const vector<corners_per_cam> filled_mv_Corners,
                                          const vector<idx_per_cam> mv_Idx,
                                          const vector<vector<cv::Mat>>& Rvec_pat2cam,
                                          const vector<vector<cv::Mat>>& Tvec_pat2cam,
                                          const vector<cv::Mat>& camPoses_mat,
                                                vector<cv::Mat>& camPoses_vec,
                                                vector<vector<cv::Mat>>& patPoses_vec)
{
    nCameras = intrins.size();
    ceres::Problem problem;
    double *cameraPoses6d   = new double[6 * nCameras];
    double *cameraIntrinsic = new double[4 * nCameras];
    int numResid;

    // initializing camera intrinsics
    for (int i = 0; i < nCameras; i++)
    {
        cameraIntrinsic[i * 4 + 0] = intrins[i].focal_x;
        cameraIntrinsic[i * 4 + 1] = intrins[i].focal_y;
        cameraIntrinsic[i * 4 + 2] = intrins[i].k1;
        cameraIntrinsic[i * 4 + 3] = intrins[i].k2;
        cout << i << " th camera intrinsics : "<<intrins[i].focal_x<<" "<<intrins[i].focal_y<<" "<<
        intrins[i].princ_x<<" "<<intrins[i].princ_y<<" "<<intrins[i].k1<<" "<<intrins[i].k2<<endl;
    }
    
    // initializing camera poses
    for (int i = 0; i < nCameras; i++){
        cout<<i<<"th cam pose: ";
        for (int j = 0; j < 6; j++){
            cameraPoses6d[i * 6 + j] = camPoses_vec[i].at<double>(j);
            cout<<cameraPoses6d[i * 6 + j]<<", ";
        }
        cout<<endl;
    }

    double **patternPoses6d= new double* [nCameras];
    for(int i=0; i<nCameras; i++)
    {
        int nImgs = mv_Idx[i].size();
        patternPoses6d[i] = new double[nImgs*6];
        for(int j=0; j<nImgs; j++)
        {
            patternPoses6d[i][j*6+0] = Rvec_pat2cam[i][j].at<double>(0);
            patternPoses6d[i][j*6+1] = Rvec_pat2cam[i][j].at<double>(1);
            patternPoses6d[i][j*6+2] = Rvec_pat2cam[i][j].at<double>(2);
            patternPoses6d[i][j*6+3] = Tvec_pat2cam[i][j].at<double>(0);
            patternPoses6d[i][j*6+4] = Tvec_pat2cam[i][j].at<double>(1);
            patternPoses6d[i][j*6+5] = Tvec_pat2cam[i][j].at<double>(2);
        }
    }

    // loop cameras/ 1st camera is the referece camera
    int nWholeImgs = filled_mv_Corners[0].size();
    int nWholePts = filled_mv_Corners[0][0].size();
    cout<<"nWholeImgs: "<<nWholeImgs<<endl;
    cout<<"nWholePts: "<<nWholePts<<endl;
    const int except_ref_cam = 0;
    int refFrameFlag;
    
    for (int ii = 0; ii < nCameras; ii++)
    {
        if(ii == except_ref_cam) refFrameFlag = 1;
        else refFrameFlag = -1;

        for(int jj = ii+1; jj < nCameras; jj++)
        {
            int partial_id_ii = -1;
            int partial_id_jj = -1;
            for (int kk = 0; kk < nWholeImgs; kk++)
            {
                int cam_ii_captured_img_kk = -1;
                int cam_jj_captured_img_kk = -1;
                if(filled_mv_Corners[ii][kk].size() > 0) { cam_ii_captured_img_kk = 1; partial_id_ii++;}
                if(filled_mv_Corners[jj][kk].size() > 0) { cam_jj_captured_img_kk = 1; partial_id_jj++;} 
                if((cam_ii_captured_img_kk + cam_jj_captured_img_kk) < 2 ) continue;
                
                for (int hh = 0; hh < nWholePts; hh++)
                {
                    cv::Point2f pp1 = cv::Point2f(filled_mv_Corners[ii][kk][hh]);
                    if(pp1.x < 1 || pp1.y < 1) continue;

                    cv::Point2f pp2 = cv::Point2f(filled_mv_Corners[jj][kk][hh]);
                    if(pp2.x < 1 || pp2.y < 1) continue;

                    cv::Point3f PP = world3d[hh];

                    ceres::CostFunction *cost_ftn = ReprojectionError::create(pp1.x, pp1.y, pp2.x, pp2.y,PP.x, PP.y, PP.z, 1920, 1080, refFrameFlag, numResid);
                    problem.AddResidualBlock(cost_ftn, new ceres::HuberLoss(2.0), cameraIntrinsic+(ii*4), cameraIntrinsic+(jj*4), 
                        patternPoses6d[ii]+(partial_id_ii*6), patternPoses6d[jj]+(partial_id_jj*6), cameraPoses6d+(ii*6), cameraPoses6d+(jj*6));
                }
            }
        }        
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR; //ceres::SPARSE_NORMAL_CHOLESKY;ceres::ITERATIVE_SCHUR; //
    //options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!(summary.termination_type == ceres::CONVERGENCE))
    {
        cout << "Bundle Adjustment Failed\n"
             << endl;
    }

    // updating intrinsics
    for (int i = 0; i < nCameras; i++)
    {
        intrins[i].focal_x = cameraIntrinsic[i * 4 + 0];
        intrins[i].focal_y = cameraIntrinsic[i * 4 + 1];
        intrins[i].k1      = cameraIntrinsic[i * 4 + 2];
        intrins[i].k2      = cameraIntrinsic[i * 4 + 3];
    }
    
    // updating camera poses
    for (int i = 0; i < nCameras; i++){
        for (int j = 0; j < 6; j++){
            camPoses_vec[i].at<double>(j) = cameraPoses6d[i * 6 + j];
        }
    }
    // updating pattern poses
    patPoses_vec.resize(nCameras);
    for(int i=0; i<nCameras; i++){
        int nImgs = mv_Idx[i].size();
        patPoses_vec[i].resize(nImgs);
        for(int j=0; j<nImgs; j++)
        {
            patPoses_vec[i][j] = (cv::Mat_<double>(6, 1)<<
                patternPoses6d[i][j*6+0], patternPoses6d[i][j*6+1], patternPoses6d[i][j*6+2],
                patternPoses6d[i][j*6+3], patternPoses6d[i][j*6+4], patternPoses6d[i][j*6+5] );
        }
    }

    for (int i = 0; i < nCameras; i++)
    {
        // refined cam poses
        cout << i << " th camera pose : ";
        for(int j=0; j<6; j++)
            cout<< camPoses_vec[i].at<double>(j) <<" ";
        cout<<endl;
    }

    for (int i = 0; i < nCameras; i++)
    {
        // refined cam poses
        cout << i << " th camera intrinsics : "<<intrins[i].focal_x<<" "<<intrins[i].focal_y<<" "<<
        intrins[i].princ_x<<" "<<intrins[i].princ_y<<" "<<intrins[i].k1<<" "<<intrins[i].k2<<endl;
        cout<<endl;
    }

    std::cout << summary.BriefReport() << endl;
    std::cout << summary.FullReport() << endl;
}
} // namespace sfmlib

using namespace sfmlib;

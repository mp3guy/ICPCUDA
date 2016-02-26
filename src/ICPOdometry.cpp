/*
 * ICPOdometry.cpp
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#include "icpcuda/ICPOdometry.h"

ICPOdometry::ICPOdometry(int width,
                         int height,
                         float cx, float cy, float fx, float fy,
                         float distThresh,
                         float angleThresh)
: lastICPError(0),
  lastICPCount(width * height),
  lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
  lastb(Eigen::Matrix<double, 6, 1>::Zero()),
  distThres_(distThresh),
  angleThres_(angleThresh),
  width(width),
  height(height),
  cx(cx), cy(cy), fx(fx), fy(fy)
{
    sumData.create(MAX_THREADS);
    outData.create(1);

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.reserve(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_g_prev_.resize(NUM_PYRS);
    nmaps_g_prev_.resize(NUM_PYRS);

    vmaps_curr_.resize(NUM_PYRS);
    nmaps_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create (pyr_rows, pyr_cols);

        vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
        nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

        vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
        nmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    }
}

ICPOdometry::~ICPOdometry()
{

}

void ICPOdometry::initICP(unsigned short * depth, const float depthCutoff)
{
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);
    initICP(depth_tmp[0], depthCutoff);
}

void ICPOdometry::initICP(const DeviceArray2D<unsigned short>& depth, const float depthCutoff)
{
    depth_tmp[0] = depth;

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_curr_[i], depthCutoff);
        createNMap(vmaps_curr_[i], nmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::initICPModel(unsigned short * depth,
                               const float depthCutoff,
                               const Eigen::Matrix4f & modelPose)
{
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);
    initICPModel(depth_tmp[0], depthCutoff, modelPose);
}

void ICPOdometry::initICPModel(const DeviceArray2D<unsigned short>& depth,
                               const float depthCutoff,
                               const Eigen::Matrix4f & modelPose)
{
    depth_tmp[0] = depth;

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_g_prev_[i], depthCutoff);
        createNMap(vmaps_g_prev_[i], nmaps_g_prev_[i]);
    }

    cudaDeviceSynchronize();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    Mat33 &  device_Rcam = device_cast<Mat33>(Rcam);
    float3& device_tcam = device_cast<float3>(tcam);

    if (modelPose != Eigen::Matrix4f::Identity())
      for(int i = 0; i < NUM_PYRS; ++i)
        tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);

    cudaDeviceSynchronize();
}

void ICPOdometry::getIncrementalTransformation(Eigen::Vector3f & trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot, int threads, int blocks)
{
    iterations[0] = 10;
    iterations[1] = 5;
    iterations[2] = 4;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    Mat33 & device_Rprev_inv = device_cast<Mat33>(Rprev_inv);
    float3& device_tprev = device_cast<float3>(tprev);

    cv::Mat resultRt = cv::Mat::eye(4, 4, CV_64FC1);

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        for(int j = 0; j < iterations[i]; j++)
        {
            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3& device_tcurr = device_cast<float3>(tcurr);

            DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
            DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

            float residual[2];

            icpStep(device_Rcurr,
                    device_tcurr,
                    vmap_curr,
                    nmap_curr,
                    device_Rprev_inv,
                    device_tprev,
                    intr(i),
                    vmap_g_prev,
                    nmap_g_prev,
                    distThres_,
                    angleThres_,
                    sumData,
                    outData,
                    A_icp.data(),
                    b_icp.data(),
                    &residual[0],
                    threads,
                    blocks);

            lastICPError = sqrt(residual[0]) / residual[1];
            lastICPCount = residual[1];

            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            lastA = dA_icp;
            lastb = db_icp;
            result = lastA.ldlt().solve(lastb);

            Eigen::Isometry3f incOdom;

            OdometryProvider::computeProjectiveMatrix(resultRt, result, incOdom);

            Eigen::Isometry3f currentT;
            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * incOdom.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();
        }
    }

    trans = tcurr;
    rot = Rcurr;
}

Eigen::MatrixXd ICPOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}

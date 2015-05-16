/*
 * ICPOdometry.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#ifndef ICPODOMETRY_H_
#define ICPODOMETRY_H_

#include "Cuda/internal.h"
#include "OdometryProvider.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <vector_types.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>

class ICPOdometry
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ICPOdometry(int width,
                    int height,
                    float cx, float cy, float fx, float fy,
                    float distThresh = 0.10f,
                    float angleThresh = sin(20.f * 3.14159254f / 180.f));

        virtual ~ICPOdometry();

        void initICP(unsigned short * depth, const float depthCutoff);

        void initICPModel(unsigned short * depth, const float depthCutoff, const Eigen::Matrix4f & modelPose);

        void getIncrementalTransformation(Eigen::Vector3f & trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot, int threads, int blocks);

        Eigen::MatrixXd getCovariance();

        float lastICPError;
        float lastICPCount;

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA;
        Eigen::Matrix<double, 6, 1> lastb;

    private:
        std::vector<DeviceArray2D<unsigned short> > depth_tmp;

        std::vector<DeviceArray2D<float> > vmaps_g_prev_;
        std::vector<DeviceArray2D<float> > nmaps_g_prev_;

        std::vector<DeviceArray2D<float> > vmaps_curr_;
        std::vector<DeviceArray2D<float> > nmaps_curr_;

        Intr intr;

        DeviceArray<jtjjtr> sumData;
        DeviceArray<jtjjtr> outData;

        static const int NUM_PYRS = 3;

        std::vector<int> iterations;

        float distThres_;
        float angleThres_;

        Eigen::Matrix<double, 6, 6> lastCov;

        const int width;
        const int height;
        const float cx, cy, fx, fy;
};

#endif /* ICPODOMETRY_H_ */

/*
 * OdometryProvider.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#ifndef ODOMETRYPROVIDER_H_
#define ODOMETRYPROVIDER_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

class OdometryProvider
{
    public:
        OdometryProvider()
        {}

        virtual ~OdometryProvider()
        {}

        static inline void computeProjectiveMatrix(cv::Mat & resultRt, const Eigen::Matrix<double, 6, 1> & result, Eigen::Isometry3f & incOdom)
        {
            cv::Mat Rt = cv::Mat::eye(4, 4, CV_64FC1);

            cv::Mat R = Rt(cv::Rect(0,0,3,3));
            cv::Mat rvec(cv::Point3d(result(3), result(4), result(5)));

            cv::Rodrigues(rvec, R);

            Rt.at<double>(0, 3) = result(0);
            Rt.at<double>(1, 3) = result(1);
            Rt.at<double>(2, 3) = result(2);

            resultRt = Rt * resultRt;

            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation;
            rotation(0, 0) = resultRt.at<double>(0, 0);
            rotation(0, 1) = resultRt.at<double>(0, 1);
            rotation(0, 2) = resultRt.at<double>(0, 2);

            rotation(1, 0) = resultRt.at<double>(1, 0);
            rotation(1, 1) = resultRt.at<double>(1, 1);
            rotation(1, 2) = resultRt.at<double>(1, 2);

            rotation(2, 0) = resultRt.at<double>(2, 0);
            rotation(2, 1) = resultRt.at<double>(2, 1);
            rotation(2, 2) = resultRt.at<double>(2, 2);

            Eigen::Vector3f translation;
            translation(0) = resultRt.at<double>(0, 3);
            translation(1) = resultRt.at<double>(1, 3);
            translation(2) = resultRt.at<double>(2, 3);

            incOdom.setIdentity();
            incOdom.rotate(rotation);
            incOdom.translation() = translation;
        }
};

#endif /* ODOMETRYPROVIDER_H_ */

#ifndef camera_parameters_h
#define camera_parameters_h

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/contrib/contrib.hpp>


// kitti image size
const int imageWidth = 1226;  // orig: 744
const int imageHeight = 370;  // orig: 480

//double kImage[9] = {4.177343016733e+002, 0, 3.715643918956e+002,
//                    0, 4.177970397634e+002, 1.960688121183e+002,
//                    0, 0, 1};


double kImage[9] = {707.0912, 0, 601.8873,  // kitti 07
                    0, 707.0912, 183.1104,
                    0, 0, 1};

//double dImage[4] = {-3.396867101163e-001, 1.309347902588e-001, -2.346791258754e-004, 2.209387016957e-004};

double dImage[4] = {0.0, 0.0, 0.0, 0.0};    // kitti

#endif


#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;

const int imagePixelNum = 1226 * 370;
CvSize imgSize = cvSize(1226, 370);
IplImage *imageCur = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
Mat imgcur(imgSize, CV_8UC1);
string path = "/media/vance/00077298000E1760/dataset/KITTI/2011_09_30/2011_09_30_drive_0027_sync/image_00/data";

ros::Publisher *imageShowPubPointer;
void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData)
{
    for (int i = 0; i < imagePixelNum; i++) {
      imageCur->imageData[i] = (char)imageData->data[i];
      if (i % 10000 == 0)
          printf("%d --- %d, %d\n", i, (int)imageData->data[i], (unsigned int)imageCur->imageData[i]);
    }
    imgcur = cv_bridge::toCvShare(imageData, "mono8")->image;
    imshow("IMG curr", imgcur);

    Mat imgMat = cvarrToMat(imageCur);
    cout << " Mat size: width " << imgMat.cols << ", height " << imgMat.rows << endl;
    for (int i = 0; i < imagePixelNum; i++) {
      if (i % 10000 == 0)
          printf("%d --- %d\n", i,  imgMat.ptr<char>(i));
    }
    imshow("IMG", imgMat);

    waitKey(100);



}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "img_convert");
  ros::NodeHandle nh;

  ROS_INFO("img_convert");

  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/kitti/camera_gray_left/image_raw", 1, imageDataHandler);

  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);
  imageShowPubPointer = &imageShowPub;

  ros::spin();

  return 0;

}

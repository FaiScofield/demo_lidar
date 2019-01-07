#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <string>

#include "cameraParameters.h"
#include "pointDefinition.h"

using namespace std;
using namespace cv;

//string path = "/media/vance/00077298000E1760/dataset/KITTI/2011_09_30/2011_09_30_drive_0027_sync/image_00/data";

bool systemInited = false;
double timeCur, timeLast;

/// 操作对象 - 当前帧和上一帧的图像
const int imagePixelNum = imageHeight * imageWidth;
CvSize imgSize = cvSize(imageWidth, imageHeight);
IplImage *imageCur = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
IplImage *imageLast = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
//Mat *imageCur = new Mat(imgSize, CV_8UC1);
//Mat *imageLast = new Mat(imgSize, CV_8UC1);

/// 可视化参数
int showCount = 0;          //
const int showSkipNum = 2;  // 可视化时要跳过的帧数
const int showDSRate = 2;   // 可视化画面比例
CvSize showSize = cvSize(imageWidth / showDSRate, imageHeight / showDSRate); // 以原图1/4大小可视化
IplImage *imageShow = cvCreateImage(showSize, IPL_DEPTH_8U, 1);
IplImage *harrisLast = cvCreateImage(showSize, IPL_DEPTH_32F, 1);

/// 相机参数矩阵
CvMat kMat = cvMat(3, 3, CV_64FC1, kImage);
CvMat dMat = cvMat(4, 1, CV_64FC1, dImage);

IplImage *mapx, *mapy;
//Mat *mapx, *mapy;

/// 区域划分
const int maxFeatureNumPerSubregion = 2; // 每个子区域的最大特征数
const int xSubregionNum = 12; // 宽度上的区域划分数
const int ySubregionNum = 8;  // 高度上的区域划分数
const int totalSubregionNum = xSubregionNum * ySubregionNum; // 总区域数
const int MAXFEATURENUM = maxFeatureNumPerSubregion * totalSubregionNum; // 总特征数量不会超过的最大值

const int xBoundary = 20; // 左右预留的边界像素量
const int yBoundary = 20; // 上下预留的边界像素量
const double subregionWidth = (double)(imageWidth - 2 * xBoundary) / (double)xSubregionNum; // 单个子区域的宽度
const double subregionHeight = (double)(imageHeight - 2 * yBoundary) / (double)ySubregionNum; // 单个子区域的高度

const double maxTrackDis = 100;
const int winSize = 15;

/// 图像特征
IplImage *imageEig, *imageTmp, *pyrCur, *pyrLast; // 图像金字塔

CvPoint2D32f *featuresCur = new CvPoint2D32f[2 * MAXFEATURENUM]; // 为何给2倍空间？
CvPoint2D32f *featuresLast = new CvPoint2D32f[2 * MAXFEATURENUM];
char featuresFound[2 * MAXFEATURENUM];
float featuresError[2 * MAXFEATURENUM];

int featuresIndFromStart = 0; // 特征点的相对第一个点的索引
int featuresInd[2 * MAXFEATURENUM] = {0}; // 所有特征点的相对索引

int totalFeatureNum = 0; // 总特征点数
int subregionFeatureNum[2 * totalSubregionNum] = {0}; // 每个子区域的特征点数，为何2倍？

/// 转为深度图？pcl格式？
pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<ImagePoint>::Ptr imagePointsLast(new pcl::PointCloud<ImagePoint>());

/// ROS相关
ros::Publisher *imagePointsLastPubPointer;
ros::Publisher *imageShowPubPointer;
cv_bridge::CvImage bridge;

/// 图像的处理
/// 1）提取fast角点 2）计算图像金字塔
void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData)
{

  // 前帧数据变后帧，当前帧数据更新
  timeLast = timeCur;
  timeCur = imageData->header.stamp.toSec() - 0.1163; // -0.1163 ? 是因为ROS消息提取的延迟而补偿？

  IplImage *imageTemp = imageLast;
  imageLast = imageCur;
  imageCur = imageTemp;


  for (int i = 0; i < imagePixelNum; i++) {
    imageCur->imageData[i] = (char)imageData->data[i];
  }

  IplImage *t = cvCloneImage(imageCur);

  // 去除当前帧图像畸变
  cvRemap(t, imageCur, mapx, mapy);
//  cvEqualizeHist(imageCur, imageCur); // 直方图均衡化
  cvReleaseImage(&t);

  // 将上一帧带有特征点的图像输出
  cvResize(imageLast, imageShow);
  cvCornerHarris(imageShow, harrisLast, 3);

  CvPoint2D32f *featuresTemp = featuresLast;
  featuresLast = featuresCur;
  featuresCur = featuresTemp;

  pcl::PointCloud<ImagePoint>::Ptr imagePointsTemp = imagePointsLast;
  imagePointsLast = imagePointsCur;
  imagePointsCur = imagePointsTemp;
  imagePointsCur->clear();

  if (!systemInited) {
    systemInited = true;
    return; // 第一帧跳过，第二帧再往下
  }

  // 对每个子区域进行特征提取，分区域有助于特征点均匀分布
  // 所有区域特征提取完毕后，特征点存在featuresLast数组内，对应的索引在featuresInd数组内
  int recordFeatureNum = totalFeatureNum;
  for (int i = 0; i < ySubregionNum; i++) {
    for (int j = 0; j < xSubregionNum; j++) {
      int ind = xSubregionNum * i + j;  // ind指向当前的subregion编号
      int numToFind = maxFeatureNumPerSubregion - subregionFeatureNum[ind];

      if (numToFind > 0) {
        int subregionLeft = xBoundary + (int)(subregionWidth * j);
        int subregionTop = yBoundary + (int)(subregionHeight * i);
        CvRect subregion = cvRect(subregionLeft, subregionTop, (int)subregionWidth, (int)subregionHeight);
        cvSetImageROI(imageLast, subregion); // 将当前的subregion设置为ROI区域

        // 在ROI中寻找好的特征点,存在featuresLast的数组内（有2倍MAXFEATURENUM的空间）
        cvGoodFeaturesToTrack(imageLast, imageEig, imageTmp, featuresLast + totalFeatureNum,
                              &numToFind, 0.1, 5.0, nullptr, 3, 1, 0.04);

        int numFound = 0;
        for(int k = 0; k < numToFind; k++) {
          // 特征点的横纵坐标是相对于子区域左上角的，这里更新为绝对坐标
          featuresLast[totalFeatureNum + k].x += subregionLeft;
          featuresLast[totalFeatureNum + k].y += subregionTop;

          // 特征点在可视化图像上的坐标
          int xInd = (featuresLast[totalFeatureNum + k].x + 0.5) / showDSRate;
          int yInd = (featuresLast[totalFeatureNum + k].y + 0.5) / showDSRate;

          // 筛选特征点
          if (((float*)(harrisLast->imageData + harrisLast->widthStep * yInd))[xInd] > 1e-7) {
            featuresLast[totalFeatureNum + numFound].x = featuresLast[totalFeatureNum + k].x;
            featuresLast[totalFeatureNum + numFound].y = featuresLast[totalFeatureNum + k].y;
            featuresInd[totalFeatureNum + numFound] = featuresIndFromStart;

            numFound++;
            featuresIndFromStart++;
          }
        }
        totalFeatureNum += numFound;
        subregionFeatureNum[ind] += numFound;

        cvResetImageROI(imageLast); // 取消ROI区域
      }
    }
  }

  // 计算图像金字塔
  cvCalcOpticalFlowPyrLK(imageLast, imageCur, pyrLast, pyrCur,
                         featuresLast, featuresCur, totalFeatureNum, cvSize(winSize, winSize),
                         3, featuresFound, featuresError,
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01), 0);

  for (int i = 0; i < totalSubregionNum; i++) {
    subregionFeatureNum[i] = 0;
  }

  ImagePoint point;
  int featureCount = 0;
  double meanShiftX = 0, meanShiftY = 0;
  for (int i = 0; i < totalFeatureNum; i++) {
    double trackDis = sqrt((featuresLast[i].x - featuresCur[i].x)
                    * (featuresLast[i].x - featuresCur[i].x)
                    + (featuresLast[i].y - featuresCur[i].y)
                    * (featuresLast[i].y - featuresCur[i].y));

    if (!(trackDis > maxTrackDis || featuresCur[i].x < xBoundary ||
      featuresCur[i].x > imageWidth - xBoundary || featuresCur[i].y < yBoundary ||
      featuresCur[i].y > imageHeight - yBoundary)) {

      // 计算当前特征点是哪个subregion中检测到的，ind是subregion的编号
      int xInd = (int)((featuresLast[i].x - xBoundary) / subregionWidth);
      int yInd = (int)((featuresLast[i].y - yBoundary) / subregionHeight);
      int ind = xSubregionNum * yInd + xInd;

      if (subregionFeatureNum[ind] < maxFeatureNumPerSubregion) {
        // 根据筛选准则将光流法匹配到的特征点进行筛选,这里featureCount是从0开始的，
        // 所以featuresCur[]和featuresLast[]只保存了邻近图像的特征点，很久之前的没有保存
        featuresCur[featureCount].x = featuresCur[i].x;
        featuresCur[featureCount].y = featuresCur[i].y;
        featuresLast[featureCount].x = featuresLast[i].x;
        featuresLast[featureCount].y = featuresLast[i].y;
        // 有些特征点被筛掉，所以这里featureCount不一定和i相等
        featuresInd[featureCount] = featuresInd[i];

        /* 这一步将图像坐标系下的特征点[u,v]，变换到了相机坐标系下，即[u,v]->[X/Z,Y/Z,1],参考《14讲》式5.5
         * 不过要注意这里加了个负号。相机坐标系默认是z轴向前，x轴向右，y轴向下，图像坐标系默认在图像的左上角，
         * featuresCur[featureCount].x - kImage[2]先将图像坐标系从左上角还原到图像中心，然后加个负号，
         * 即将默认相机坐标系的x轴负方向作为正方向，y轴同理。所以此时相机坐标系z轴向前，x轴向左，y轴向上
         */
        point.u = -(featuresCur[featureCount].x - kImage[2]) / kImage[0];
        point.v = -(featuresCur[featureCount].y - kImage[5]) / kImage[4];
        point.ind = featuresInd[featureCount];
        imagePointsCur->push_back(point);

        if (i >= recordFeatureNum) {
          point.u = -(featuresLast[featureCount].x - kImage[2]) / kImage[0];
          point.v = -(featuresLast[featureCount].y - kImage[5]) / kImage[4];
          imagePointsLast->push_back(point);
        }

        meanShiftX += fabs((featuresCur[featureCount].x - featuresLast[featureCount].x) / kImage[0]);
        meanShiftY += fabs((featuresCur[featureCount].y - featuresLast[featureCount].y) / kImage[4]);

        featureCount++;
        subregionFeatureNum[ind]++; // subregionFeatureNum是根据当前帧与上一帧的特征点匹配数目来计数的
      }
    }
  }
  totalFeatureNum = featureCount;
  meanShiftX /= totalFeatureNum;
  meanShiftY /= totalFeatureNum;

  sensor_msgs::PointCloud2 imagePointsLast2;
  pcl::toROSMsg(*imagePointsLast, imagePointsLast2);
  imagePointsLast2.header.frame_id = "camera";
  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLastPubPointer->publish(imagePointsLast2);

  // 每隔两帧图像才输出一副图像显示
  showCount = (showCount + 1) % (showSkipNum + 1);
  if (showCount == showSkipNum) {
    Mat imageShowMat = cvarrToMat(imageShow);
    bridge.image = imageShowMat;
    bridge.encoding = "mono8";
    bridge.header.stamp = ros::Time().fromSec(timeLast);
    bridge.header.frame_id = "camera";
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    imageShowPubPointer->publish(imageShowPointer);
    cout << "pub a msg.\n";
  }
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "featureTracking");
  ros::NodeHandle nh;

  // 计算图像去畸变的投影变换，存在mapx和mapy里，在imageDataHandler()函数里进行去畸变处理
  mapx = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
  mapy = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
  cvInitUndistortMap(&kMat, &dMat, mapx, mapy);


  CvSize subregionSize = cvSize((int)subregionWidth, (int)subregionHeight);
  imageEig = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);
  imageTmp = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);

  CvSize pyrSize = cvSize(imageWidth + 8, imageHeight / 3);
  pyrCur = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);
  pyrLast = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);

//  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/image/raw", 1, imageDataHandler);
  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/kitti/camera_gray_left/image_raw", 1, imageDataHandler);


  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
  imagePointsLastPubPointer = &imagePointsLastPub;

  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);
  imageShowPubPointer = &imageShowPub;

  ros::spin();

  return 0;
}

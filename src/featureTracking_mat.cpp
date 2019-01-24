#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <string>
#include <opencv2/video/tracking.hpp>
#include <chrono>

#include "cameraParameters.h"
#include "pointDefinition.h"

using namespace std;
using namespace cv;


class FeatureTracking {

private:
    /// ROS相关
    ros::NodeHandle nh;
    ros::Subscriber imageDataSub;
    ros::Publisher imagePointsLastPub;
    ros::Publisher imageShowPub;

    cv_bridge::CvImage bridge;
    pcl::PointCloud<ImagePoint>::Ptr imagePointsCur;
    pcl::PointCloud<ImagePoint>::Ptr imagePointsLast;


    bool systemInited, isSecondFrame;
    double timeLast, timeCur;

    int detectionCount;
    const int detectionSkipNum = 3;

    /// 可视化参数
    int showCount;

    /// const
//    const int imagePixelNum = imageHeight * imageWidth;

    const size_t maxFeatureNumPerSubregion = 20; // 每个子区域的最大特征数
    const size_t xSubregionNum = 8; // 宽度上的区域划分数, for kitti
    const size_t ySubregionNum = 3; // 高度上的区域划分数
    const size_t totalSubregionNum = xSubregionNum * ySubregionNum; // 总区域数
//    const int MAXFEATURENUM = maxFeatureNumPerSubregion * totalSubregionNum; // 总特征数量不会超过的最大值
    const int xBoundary = 20, yBoundary = 20; // 左右和上下预留的边界像素量
    const double subregionWidth = (imageWidth - 2 * xBoundary) / static_cast<double>(xSubregionNum); // 单个子区域的宽度
    const double subregionHeight = (imageHeight - 2 * yBoundary) / static_cast<double>(ySubregionNum); // 单个子区域的高度
    const double maxTrackDis = 100.0;  // 光流最大距离

    const int showSkipNum = 2;  // 可视化时要跳过的帧数
    const int showDSRate = 1;   // 可视化画面比例

    /// 操作对象 - 当前帧和上一帧的图像
    Size imgSize, showSize; // 以原图1/4大小可视化
    Mat imageCur, imageLast, imageShow, harrisLast;
    Mat kMat, dMat, mapx, mapy; // 相机参数矩阵
    Mat imageEig, imageTmp, pyrCur, pyrLast; // 图像金字塔

    vector<Point2f> featuresCur, featuresLast; // 存放两帧的特征点
//    vector<Point2f> featuresSub;    // 存放图像某个子区域内检测到的特征点
    vector<unsigned char> featuresStatus; // 光流追踪到的特征点的标志
    vector<float> featuresError;    // 光流追踪到的特征点的误差

    size_t featuresIndFromStart; // 特征点的相对第一个点的索引
    size_t totalFeatureNum; // 总特征点数
    vector<size_t> featuresInd; // 所有特征点的相对索引
    vector<size_t> subregionFeatureNum; // 每个子区域的特征点数

public:
    FeatureTracking() : nh("~") {
        imageDataSub = nh.subscribe<sensor_msgs::Image>("/kitti/camera_gray_left/image_raw", 1, &FeatureTracking::imageDataHandler, this);

        imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
        imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);

        initializationValue();
    }

    void initializationValue() {
        imagePointsCur.reset(new pcl::PointCloud<ImagePoint>());
        imagePointsLast.reset(new pcl::PointCloud<ImagePoint>());

        systemInited = false;
        isSecondFrame = true;
        timeLast = timeCur = 0.0;

        detectionCount = 0;
        imgSize = Size(imageWidth, imageHeight);

        /// 可视化参数
        showCount = 0;
        showSize = cvSize(imageWidth / showDSRate, imageHeight / showDSRate); // 以原图1/4大小可视化

        /// 相机参数矩阵
        kMat = Mat(3, 3, CV_64FC1, kImage);
        dMat = Mat(4, 1, CV_64FC1, dImage);

        featuresIndFromStart = 0; // 特征点的相对第一个点的索引
        totalFeatureNum = 0; // 一帧图像的总特征点数
        subregionFeatureNum.resize(static_cast<size_t>(totalSubregionNum), 0); // 每个子区域的特征点数

        // 计算图像去畸变的投影变换，存在mapx和mapy里，在imageDataHandler()函数里进行去畸变处理
        Size imageSize = Size(imageWidth, imageHeight);
        mapx.create(imageSize, CV_32FC1);
        mapy.create(imageSize, CV_32FC1);
        initUndistortRectifyMap(kMat, dMat, Mat(), kMat, imageSize, CV_32FC1, mapx, mapy);
    }

    /// 图像的处理，1）提取fast角点 2）计算图像金字塔
    void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData) {
        auto t1 = chrono::steady_clock::now();

        // 前帧数据变后帧，当前帧数据更新
        timeLast = timeCur;
        timeCur = imageData->header.stamp.toSec() - 0.1163; // -0.1163 ? 是因为ROS消息提取的延迟而补偿？

        cv_bridge::CvImageConstPtr imageDataCv = cv_bridge::toCvShare(imageData, "mono8");

        // 首帧的处理
        if (!systemInited) {
            remap(imageDataCv->image, imageCur, mapx, mapy, CV_INTER_LINEAR); // 去除首帧图像畸变
            computeFeatures(&imageCur, featuresCur); // 提取首帧特征点，有数据更新

//            if (featuresLast.size() > 0.1 * MAXFEATURENUM) {
                systemInited = true; // 首帧特征点数量不至太少则系统得以初始化完毕
                cout << "***** system is ready! \n";
//            }

            return;
        }

        // 前后帧数据交换
        imageLast = imageCur;
        featuresLast.swap(featuresCur);
        featuresCur.clear();
        cout << "featuresLast size: " << featuresLast.size() << endl;
        pcl::PointCloud<ImagePoint>::Ptr imagePointsTemp = imagePointsLast;
        imagePointsLast = imagePointsCur;
        imagePointsCur = imagePointsTemp;
        if (!imagePointsCur->empty()) imagePointsCur->clear();

        remap(imageDataCv->image, imageCur, mapx, mapy, CV_INTER_LINEAR); // 去除当前帧图像畸变
        computeFeatures(&imageCur, featuresCur); // 提取当前帧特征点
        cout << "computeFeatures: " << totalFeatureNum << endl;

        // 计算上一帧显示图像的角点，并将将其输出
        resize(imageLast, imageShow, showSize);
//        cornerHarris(imageShow, harrisLast, 3, 3, 0.04);

        // 每隔两帧图像才计算一次
//        int recordFeatureNum = totalFeatureNum;
//        detectionCount = (detectionCount + 1) % (detectionSkipNum + 1);
//        if (detectionCount == detectionSkipNum) {
//        }

        // 计算图像金字塔
        if (totalFeatureNum > 0) {
//            Mat featuresLastMat(1, totalFeatureNum, CV_32FC2, (void*)&featuresLast[0]);

            calcOpticalFlowPyrLK(imageLast, imageCur, featuresLast, featuresCur, featuresStatus, featuresError);
            if (featuresCur.size() > 0 && featuresCur.size() == featuresStatus.size()) {
                totalFeatureNum = featuresCur.size();  // 更新成光流金字塔检测到的特征点数
            } else {
                totalFeatureNum = 0;
            }
            cout << "totalFeatureNum after calcOpticalFlowPyrLK: " << totalFeatureNum << endl;
        }

        subregionFeatureNum.clear();
        subregionFeatureNum.resize(static_cast<size_t>(totalSubregionNum), 0); // 待更新成匹配成功的特征点数


        // 特征点分区域进行匹配并求相机坐标系下的坐标
        ImagePoint point;
        size_t featureCount = 0;
        for (size_t i = 0; i < totalFeatureNum; ++i) {
            float trackDis_square = (featuresLast[i].x - featuresCur[i].x) * (featuresLast[i].x - featuresCur[i].x)
                + (featuresLast[i].y - featuresCur[i].y) * (featuresLast[i].y - featuresCur[i].y);

            if (featuresStatus[i] && !(trackDis_square > maxTrackDis * maxTrackDis ||
                featuresCur[i].x < xBoundary || featuresCur[i].x > imageWidth - xBoundary ||
                featuresCur[i].y < yBoundary || featuresCur[i].y > imageHeight - yBoundary)) {

                // 计算当前特征点是哪个subregion中检测到的，ind是subregion的编号
                size_t xInd = static_cast<size_t>((featuresLast[i].x - xBoundary) / subregionWidth);
                size_t yInd = static_cast<size_t>((featuresLast[i].y - yBoundary) / subregionHeight);
                size_t ind = xSubregionNum * yInd + xInd;

                if (subregionFeatureNum[ind] < maxFeatureNumPerSubregion) {
                    // 根据筛选准则将光流法匹配到的特征点进行筛选,这里featureCount是从0开始的，
                    // 所以featuresCur[]和featuresLast[]只保存了邻近图像的特征点，很久之前的没有保存
                    // 有些特征点被筛掉，所以这里featureCount不一定和i相等
                    featuresCur[featureCount] = featuresCur[i];
                    featuresLast[featureCount] = featuresLast[i];
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

//                    if (i >= totalFeatureNum) {
//                        point.u = -(featuresLast[featureCount].x - kImage[2]) / kImage[0];
//                        point.v = -(featuresLast[featureCount].y - kImage[5]) / kImage[4];
//                        imagePointsLast->push_back(point);
//                    }

                    featureCount++;
                    subregionFeatureNum[ind]++;
                }
            }
        }

        // 丢弃掉未匹配上的特征点
        totalFeatureNum = featureCount;
        featuresCur.resize(totalFeatureNum);
        featuresLast.resize(totalFeatureNum);
        featuresInd.resize(totalFeatureNum);
        cout << "final valid features size: " << totalFeatureNum << endl;

        publishImagePointsLast();

//        showCount = (showCount + 1) % (showSkipNum + 1);
//        if (showCount == showSkipNum)
            publishShowImage();

        auto t2 = chrono::steady_clock::now();
        double dt = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        cout << " - feature tracking cost time: " << dt << endl;
    }

    /// 为图像计算特征点，更新了 totalFeatureNum 和 subregionFeatureNum 两个变量的值
    void computeFeatures(const Mat* image, vector<Point2f>& features) {
        featuresIndFromStart = totalFeatureNum = 0;
        subregionFeatureNum.clear();
        subregionFeatureNum.resize(static_cast<size_t>(totalSubregionNum), 0);
        if (!featuresInd.empty()) featuresInd.clear();
        if (!features.empty())    features.clear();

        // 对每个子区域进行特征提取，分区域有助于特征点均匀分布
        // 所有区域特征提取完毕后，特征点存在featuresLast数组内，对应的索引在featuresInd数组内
        vector<Point2f> featuresSub; // 存放图像某个子区域内检测到的特征点
        for (size_t i = 0; i < ySubregionNum; i++) {
            for (size_t j = 0; j < xSubregionNum; j++) {
                if (!featuresSub.empty()) featuresSub.clear();

                size_t ind = xSubregionNum * i + j;  // ind指向当前的subregion编号
                size_t numToFind = maxFeatureNumPerSubregion/* - subregionFeatureNum[ind]*/;

                if (numToFind > maxFeatureNumPerSubregion / 5.0) {
                    int subregionLeft = xBoundary + static_cast<int>(subregionWidth * j);
                    int subregionTop = yBoundary + static_cast<int>(subregionHeight * i);
                    Rect subregion = Rect(subregionLeft, subregionTop, static_cast<int>(subregionWidth), static_cast<int>(subregionHeight));
                    Mat mask = Mat::zeros(imgSize, CV_8UC1);
                    mask(subregion).setTo(255); // 将当前的subregion设置为图像的掩模(ROI)

                    // 在ROI中寻找好的特征点,存在featuresSub内
                    goodFeaturesToTrack(*image, featuresSub, numToFind, 0.1, 10.0, mask, 4, 1, 0.04);

                    size_t numFound = 0;
                    for(size_t k = 0; k < featuresSub.size(); k++) {
                        // 特征点的横纵坐标是相对于子区域左上角的，这里更新为绝对坐标
                        featuresSub[k].x += subregionLeft;
                        featuresSub[k].y += subregionTop;

                        // 特征点在可视化图像上的坐标
                        int xInd = static_cast<int>((featuresSub[k].x/* + 0.5*/) / showDSRate);
                        int yInd = static_cast<int>((featuresSub[k].y/* + 0.5*/) / showDSRate);

                        // 筛选特征点
                        if (xInd >= 0 && xInd < imageWidth &&
                            yInd >= 0 && yInd < imageHeight
//                            && harrisLast.at<float>(yInd, xInd) > 1e-7
                            ) {

                            features.push_back(featuresSub[k]);
                            featuresInd.push_back(featuresIndFromStart);

                            numFound++;
                            featuresIndFromStart++;
                        }
                    }
                    totalFeatureNum += numFound;
                    subregionFeatureNum[ind] += numFound;
//                    cout << "subregion " << ind << " find good features: " << numFound
//                         << "/" << numToFind << endl;
                }
            }
        }
    }

    /// 特征点云发布
    void publishImagePointsLast() {
        sensor_msgs::PointCloud2 imagePointsLast2;
        pcl::toROSMsg(*imagePointsLast, imagePointsLast2);
        imagePointsLast2.header.frame_id = "camera";
        imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
        imagePointsLastPub.publish(imagePointsLast2);
        cout << "pub a point msg.\n";
    }

    /// 图像显示消息发布
    void publishShowImage() {
        cvtColor(imageShow, imageShow, COLOR_GRAY2BGR);
        vector<KeyPoint> featuresPub;
        KeyPoint::convert(featuresLast, featuresPub);
        drawKeypoints(imageLast, featuresPub, imageShow, Scalar(0,255,0));
        bridge.image = imageShow;
        bridge.encoding = "bgr8";
//        bridge.header.stamp = ros::Time().fromSec(timeLast);
        bridge.header.stamp = ros::Time::now();
        bridge.header.frame_id = "camera";
        sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
        imageShowPub.publish(imageShowPointer);
        cout << "pub a image msg.\n";

        // test
//        imshow("show", imageShow);
//        waitKey(100);
    }

    /// 输出参数信息
    void print_node_informations() {
        printf("--> Node name: FeatureTracking\n");
        printf("  -> Image size parameter: %d x %d\n", imageWidth, imageHeight);
        printf("  -> Sub regin size: %4f x %4f\n", subregionWidth, subregionHeight);
        printf("  -> Sub regin number: %u x %u = %u\n", xSubregionNum, ySubregionNum, totalSubregionNum);
    }
};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "featureTracking");
    ros::NodeHandle nh;

    FeatureTracking ft;
    ft.print_node_informations(); // 参数输出

    ros::spin();

    return 0;
}

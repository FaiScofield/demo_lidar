#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include <nav_msgs/Odometry.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

const double PI = 3.1415926;
const double rad2deg = 180 / PI;
const double deg2rad = PI / 180;

double timeOdomBefBA;
double timeOdomAftBA;

double rollRec, pitchRec, yawRec;
double txRec, tyRec, tzRec;



ros::Publisher *voData2PubPointer = NULL;
tf::TransformBroadcaster *tfBroadcaster2Pointer = NULL;
nav_msgs::Odometry voData2;
tf::StampedTransform voDataTrans2;


void voDataHandler(const nav_msgs::Odometry::ConstPtr& voData)
{
  if (fabs(timeOdomBefBA - timeOdomAftBA) < 0.005) {

    geometry_msgs::Quaternion geoQuat = voData->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(rollRec, pitchRec, yawRec);

    txRec = voData->pose.pose.position.x;
    tyRec = voData->pose.pose.position.y;
    tzRec = voData->pose.pose.position.z;



    geoQuat = tf::createQuaternionMsgFromRollPitchYaw(rollRec, pitchRec, yawRec);

    // 所有在tf中的应用，例如getRPY（）和createQuaternionMsgFromRollPitchYaw（）等都是在一般坐标系下进行的，所以想把voData转换到一般坐标系，只要按顺序将四元数按符号对应就可以了
    // 上面voData传进来的四元数geoQuat.x和geoQuat.y加了负号，所以pitchRec和yawRec是带负号的，因此将这些欧拉角重新转换为四元数后geoQuat.y和geoQuat.z是带负号的
    // txRec~tzRec是指实际坐标系下的位移量（也就是z朝前，y向上，x向左），因此对应着转换到一般坐标系（x向前，y向左，z向上），tzRec对应着一般坐标系x轴的位移，txRec对应着一般坐标系y轴的位移
    voData2.header.stamp = voData->header.stamp;
    voData2.pose.pose.orientation.x = -geoQuat.y;
    voData2.pose.pose.orientation.y = -geoQuat.z;
    voData2.pose.pose.orientation.z = geoQuat.x;
    voData2.pose.pose.orientation.w = geoQuat.w;
    voData2.pose.pose.position.x = txRec;
    voData2.pose.pose.position.y = tyRec;
    voData2.pose.pose.position.z = tzRec;
    voData2PubPointer->publish(voData2);

    voDataTrans2.stamp_ = voData->header.stamp;
    voDataTrans2.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    voDataTrans2.setOrigin(tf::Vector3(txRec, tyRec, tzRec));
    tfBroadcaster2Pointer->sendTransform(voDataTrans2);
  }
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "transformMaintenance");
  ros::NodeHandle nh;

  ros::Subscriber voDataSub = nh.subscribe<nav_msgs::Odometry>
                              ("/cam_to_init", 1, voDataHandler);


  ros::Publisher voData2Pub = nh.advertise<nav_msgs::Odometry> ("/cam2_to_init", 1);
  voData2PubPointer = &voData2Pub;
  voData2.header.frame_id = "/camera_init";
  voData2.child_frame_id = "/camera2";

  tf::TransformBroadcaster tfBroadcaster2;
  tfBroadcaster2Pointer = &tfBroadcaster2;
  voDataTrans2.frame_id_ = "/camera_init";
  voDataTrans2.child_frame_id_ = "/camera2";

  ros::spin();

  return 0;
}

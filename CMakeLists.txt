cmake_minimum_required(VERSION 2.8.3)
project(demo_lidar)

set(CMAKE_BUILD_TYPE Debug)
#set(BUILD_STATIC_LIBS OFF)
#set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  pcl_ros
)

#set(OpenCV_DIR "/home/vance/softwares/opencv-3.3.1/build")
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(fast REQUIRED)

message(STATUS "OpenCV version :${OpenCV_VERSION}")
message(STATUS "PCL version:${PCL_VERSION}")
###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  pcl_ros
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${fast_INCLUDE_DIRS}

#  /usr/local/include # Location when using 'make system_install'
#  /usr/include       # More usual location (e.g. when installing using a package)
)

## Specify additional locations for library files


## Declare executables
add_executable(demo_featureTracking src/featureTracking.cpp)
add_executable(demo_featureTracking_mat src/featureTracking_mat.cpp)
#add_executable(demo_featureTracking_fast src/featureTracking_fast.cpp)
#add_executable(demo_featureTracking_ocl src/featureTracking_ocl.cpp)
add_executable(demo_visualOdometry src/visualOdometry.cpp)
add_executable(demo_processDepthmap src/processDepthmap.cpp)
add_executable(demo_transformMaintenance src/transformMaintenance.cpp)
add_executable(demo_registerPointCloud src/registerPointCloud.cpp)
add_executable(demo_imageConvert src/img_convert.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(demo_featureTracking ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(demo_featureTracking_mat ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
#target_link_libraries(demo_featureTracking_fast ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${fast_LIBRARIES})
#target_link_libraries(demo_featureTracking_ocl ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(demo_visualOdometry ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(demo_processDepthmap ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(demo_transformMaintenance ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(demo_registerPointCloud ${catkin_LIBRARIES})
target_link_libraries(demo_imageConvert ${catkin_LIBRARIES} ${OpenCV_LIBS})

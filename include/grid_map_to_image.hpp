/*
 *
 *  DEPRECATED
 *
*/

#ifndef GRID_MAP_TO_IMAGE_H_
#define GRID_MAP_TO_IMAGE_H_

// C++
#include <iostream>
#include <fstream>
// ROS
#include <ros/ros.h>
// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* sloppy technique... I know... but here it is harmless */
using namespace grid_map;
using namespace ros;

/* source: https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c */
bool is_file_exist(const char *fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

namespace grid_map_to_image {

  class GridMapToImage {
  protected:
    /* class private variable-members */
    ros::NodeHandle nodeHandle_;
    ros::Subscriber subscriber_;
    /* our traversability map, for easy access from any function */
    grid_map::GridMap map_;
    /* callback function */
    void topicCallback(const grid_map_msgs::GridMap& map_msg) {
      grid_map::GridMapRosConverter::fromMessage(map_msg, map_);

      const bool useTransparency = false;

      // Convert to CV image.
      cv::Mat originalImage;
      if (useTransparency) {
        // Note: The template parameters have to be set based on your encoding
        // of the image. For 8-bit images use `unsigned char`.
        GridMapCvConverter::toImage<unsigned short, 4>(map_, "elevation", CV_16UC4, originalImage);
      } else {
        GridMapCvConverter::toImage<unsigned short, 1>(map_, "elevation", CV_16UC1, originalImage);
      }

      /* save image and then display it */
      if(!is_file_exist("./originalImage1.png"))
        cv::imwrite("./originalImage1.png", originalImage);
      else if(!is_file_exist("./originalImage2.png"))
        cv::imwrite("./originalImage2.png", originalImage);
      else if(!is_file_exist("./originalImage3.png"))
        cv::imwrite("./originalImage3.png", originalImage);
      else if(!is_file_exist("./originalImage4.png"))
        cv::imwrite("./originalImage4.png", originalImage);
      else if(!is_file_exist("./originalImage5.png"))
        cv::imwrite("./originalImage5.png", originalImage);
      else if(!is_file_exist("./originalImage6.png"))
        cv::imwrite("./originalImage6.png", originalImage);
      else if(!is_file_exist("./originalImage7.png"))
        cv::imwrite("./originalImage7.png", originalImage);
      else if(!is_file_exist("./originalImage8.png"))
        cv::imwrite("./originalImage8.png", originalImage);
      else
        cv::imwrite("./originalImage.png", originalImage);

      cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
      cv::imshow("Image", originalImage);
      cv::waitKey(0);
    }

  public:
    GridMapToImage(ros::NodeHandle& nodeHandle) : nodeHandle_(nodeHandle) {
      subscriber_ = nodeHandle_.subscribe("/traversability_estimation/traversability_map", 1, &GridMapToImage::topicCallback, this);
    }
  };
}

#endif

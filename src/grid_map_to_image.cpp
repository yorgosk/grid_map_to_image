/*
 *
 *  DEPRECATED
 *
*/

#include <../include/grid_map_to_image.hpp>

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "grid_map_to_image");
  ros::NodeHandle nodeHandle("~");

  grid_map_to_image::GridMapToImage gridMapToImage(nodeHandle);

  ros::spin();

  return 0;
}

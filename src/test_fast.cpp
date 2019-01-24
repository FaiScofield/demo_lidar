#include <vector>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fast/fast.h>

using namespace std;
using namespace cv;

int main (int argc, char * argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <image_file>\n" ;
        return -1;
    }

   const int n_trials = 1000;
   std::vector<fast::fast_xy> corners;
   cv::Mat img = cv::imread(std::string(argv[1]), 0);
   if (img.empty()) {
       cerr << "Failed to load image at: " << argv[1] << endl;
       return -1;
   }
   cv::Mat downSampled;
   cv::resize(img, downSampled, cv::Size(752, 480));
   img = downSampled;

   printf("\nTesting PLAIN version\n");
   double time_accumulator = 0;
   for (int i = 0; i < n_trials; ++i) {
      corners.clear();
      double t = (double)cv::getTickCount();
      fast::fast_corner_detect_10_sse2((fast::fast_byte *)(img.data), img.cols, img.rows, img.cols, 75, corners);
      time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
   }
   printf("PLAIN took %f ms (average over %d trials).\n", time_accumulator/((double)n_trials)*1000.0, n_trials );
   printf("PLAIN version extracted %zu features.\n", corners.size());


}

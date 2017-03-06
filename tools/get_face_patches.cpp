/*
* @Author: vasezhong
* @Date:   2017-03-06 15:39:15
* @Last Modified by:   vasezhong
* @Last Modified time: 2017-03-06 18:45:16
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;

DEFINE_bool(gray, false, "");
DEFINE_int32(resize_width, 0, "");
DEFINE_int32(resize_height, 0, "");
DEFINE_int32(left_idx, -1, "");
DEFINE_int32(top_idx, -1, "");
DEFINE_int32(bottom_idx, -1, "");
DEFINE_int32(right_idx, -1, "");

int get_patch_size(const int& left_idx, const int& top_idx,
    const int& right_idx, const int& bottom_idx,
    const int& img_width, const int& img_height) {

  CHECK((left_idx >= 0 && left_idx < img_width)
      && (right_idx >= 0 && right_idx < img_width)
      && (top_idx >= 0 && top_idx < img_height)
      && (bottom_idx >= 0 && bottom_idx < img_height)) << "Crop must be in the origin image.";

  int max_val = std::max(right_idx - left_idx, bottom_idx - top_idx);
  max_val = std::min(max_val, img_width - left_idx);
  max_val = std::min(max_val, img_height - top_idx);
  return max_val;
}

std::string int_to_string(const int& number) {
    std::ostringstream oss;
    oss << number;
    return oss.str();
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Get patch from a set of images.\n"
        "Usage:\n"
        "    get_face_patches [FLAGS] LISTFILE OUTFOLDER\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 3) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/get_face_patches");
        return 1;
    }

    const bool is_color = !FLAGS_gray;
    const int resize_width = std::max<int>(0, FLAGS_resize_width);
    const int resize_height = std::max<int>(0, FLAGS_resize_height);
    const int left_idx = FLAGS_left_idx;
    const int top_idx = FLAGS_top_idx;
    const int bottom_idx = FLAGS_bottom_idx;
    const int right_idx = FLAGS_right_idx;

    std::ifstream infile(argv[1]);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    LOG(INFO) << "A total of " << lines.size() << " images.";

    std::string outfolder(argv[2]);
    for (int line_id = 0; line_id < lines.size(); ++line_id) {
        cv::Mat cv_img_origin = ReadImageToCVMat(lines[line_id], is_color);
        CHECK(cv_img_origin.data) << "Could not load " << lines[line_id];

        if (left_idx >= 0 && top_idx >= 0 && right_idx >= 0 && bottom_idx >= 0) {
            const int img_width = cv_img_origin.cols;
            const int img_height = cv_img_origin.rows;
            const int patch_size = get_patch_size(left_idx, top_idx, right_idx, bottom_idx, img_width, img_height);
            cv::Rect roi(left_idx, top_idx, patch_size, patch_size);
            cv_img_origin = cv_img_origin(roi);

            LOG(INFO) << "Patch size: " << patch_size;
        }

        cv::Mat cv_img;
        cv::resize(cv_img_origin, cv_img, cv::Size(resize_width, resize_height));
        CHECK(cv_img.data) << "Could not get patch from " << lines[line_id];

        std::string filename(outfolder + "/" + std::to_string(line_id) + ".jpg");
        cv::imwrite(filename, cv_img);
    }

#else
    LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif
    return 0;
}
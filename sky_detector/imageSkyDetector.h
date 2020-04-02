/************************************************
* Author: MaybeShewill-CV
* File: imageSkyDetector.h
* Date: 18-6-28 上午10:28
************************************************/

// 天空区域检测类

#ifndef PANO_IMAGE_QUALITY_CHECK_IMAGESKYDECTOR_H
#define PANO_IMAGE_QUALITY_CHECK_IMAGESKYDECTOR_H

#include <string>

#include <opencv2/opencv.hpp>

namespace sky_detector {

class SkyAreaDetector {
public:
    SkyAreaDetector() = default;
    ~SkyAreaDetector() = default;

    SkyAreaDetector(const SkyAreaDetector&);
    SkyAreaDetector& operator=(const SkyAreaDetector&);

    void detect(const std::string &image_file_path,
                const std::string &output_path);

    void batch_detect(const std::string &image_dir, const std::string &output_dir);

private:
    cv::Mat _src_img; // 原始路淘图像

    double f_thres_sky_max = 600;
    double f_thres_sky_min = 5;
    double f_thres_sky_search_step = 5;

    /***加载原始图像***/
    // 加载原始图像
    bool load_image(const std::string &image_file_path);

    /***检测图像天空区域***/
    // 提取图像天空区域
    bool extract_sky(const cv::Mat &src_image, cv::Mat &sky_mask);
    // 提取图像梯度信息
    void extract_image_gradient(const cv::Mat &src_image, cv::Mat &gradient_image);
    // 利用能量函数优化计算计算天空边界线
    std::vector<int> extract_border_optimal(const cv::Mat &src_image);
    // 计算天空边界线
    std::vector<int> extract_border(const cv::Mat &gradient_info_map, double thresh);
    // 改善天空边界线
    std::vector<int> refine_border(const std::vector<int> &border,  const cv::Mat &src_image);
    // 计算天空图像能量函数
    double calculate_sky_energy(const std::vector<int> &border, const cv::Mat &src_image);
    // 判断图像是否存在天空区域
    bool has_sky_region(const std::vector<int> &border, double thresh_1,
                        double thresh_2, double thresh_3);
    // 判断是否部分是天空区域
    bool has_partial_sky_region(const std::vector<int> &border, double thresh_1);
    // 显示天空区域(用于debug)
    void display_sky_region(const cv::Mat &src_image, const std::vector<int> &border,
                            cv::Mat &sky_image);
    // 制作天空掩码图
    cv::Mat make_sky_mask(const cv::Mat &src_image, const std::vector<int> &border, int type=1);
};
}

#endif //PANO_IMAGE_QUALITY_CHECK_IMAGESKYDECTOR_H

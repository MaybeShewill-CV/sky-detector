/************************************************
* Author: MaybeShewill-CV
* File: imageSkyDetector.cpp
* Date: 18-6-28 上午10:28
************************************************/

#include "imageSkyDetector.h"

#include <fstream>
#include <chrono>

#include <glog/logging.h>

#include <file_system_processor.h>

namespace sky_detector {

/***
 * 类复制构造函数
 * @param _SkyAreaDetector
 */
SkyAreaDetector::SkyAreaDetector(const SkyAreaDetector &_SkyAreaDetector) {
    this->f_thres_sky_max = _SkyAreaDetector.f_thres_sky_max;
    this->f_thres_sky_min = _SkyAreaDetector.f_thres_sky_min;
    this->f_thres_sky_search_step = _SkyAreaDetector.f_thres_sky_search_step;
}
/***
 * 类复制构造函数
 * @param _SkyAreaDetector
 * @return
 */
SkyAreaDetector& SkyAreaDetector::operator=(const SkyAreaDetector &_SkyAreaDetector) {
    this->f_thres_sky_max = _SkyAreaDetector.f_thres_sky_max;
    this->f_thres_sky_min = _SkyAreaDetector.f_thres_sky_min;
    this->f_thres_sky_search_step = _SkyAreaDetector.f_thres_sky_search_step;

    return *this;
}

/***
 * 读取图像文件
 * @param image_file_path
 * @return
 */
bool SkyAreaDetector::load_image(const std::string &image_file_path) {
    if (!file_processor::FileSystemProcessor::is_file_exist(image_file_path)) {
        LOG(ERROR) << "图像文件: " << image_file_path << "不存在" << std::endl;
        return false;
    }

    _src_img = cv::imread(image_file_path, CV_LOAD_IMAGE_UNCHANGED);

//    cv::resize(_src_img, _src_img, cv::Size(_src_img.size[1] * 4, _src_img.size[0] * 4));

    if (_src_img.empty() || !_src_img.data) {
        LOG(ERROR) << "图像文件: " << image_file_path << "读取失败" << std::endl;
        return false;
    }

    return true;
}

/***
 * 提取图像天空区域
 * @param bgrimg
 * @param skybinaryimg
 * @param horizonline
 * @return
 */
bool SkyAreaDetector::extract_sky(const cv::Mat &src_image, cv::Mat &sky_mask) {

    int image_height = src_image.size[0];
    int image_width = src_image.size[1];

    std::vector<int> sky_border_optimal = extract_border_optimal(src_image);

    if (!has_sky_region(sky_border_optimal, image_height / 30, image_height / 4, 2)) {
#ifdef DEBUG
        LOG(INFO) << "没有提取到天空区域" << std::endl;
#endif
        return false;
    }

#ifdef DEBUG
    cv::Mat sky_image;
    display_sky_region(src_image, optimized_border, sky_image);
    cv::imwrite("sky.jpg", sky_image);
    cv::imshow("sky image without refine", sky_image);
    cv::waitKey();
#endif

    if (has_partial_sky_region(sky_border_optimal, image_width / 3)) {
        std::vector<int> border_new = refine_border(sky_border_optimal, src_image);
        sky_mask = make_sky_mask(src_image, border_new);
#ifdef DEBUG
        display_sky_region(src_image, optimized_border, sky_image);
        cv::imshow("sky image with refine", sky_image);
        cv::waitKey();
#endif
        return true;
    }

    sky_mask = make_sky_mask(src_image, sky_border_optimal);

    return true;
}

/***
 * 检测图像天空区域接口
 * @param image_file_dir
 * @param check_output_dir
 */
void SkyAreaDetector::detect(const std::string &image_file_path, const std::string &output_path) {

    LOG(INFO) << "开始检测图像: " << image_file_path;

    // 加载图像
    load_image(image_file_path);

    // 提取图像天空区域
    cv::Mat sky_mask;
    extract_sky(_src_img, sky_mask);

    // 制作掩码输出
    cv::Mat sky_image;

    int image_height = _src_img.size[0];
    int image_width = _src_img.size[1];

    cv::Mat sky_image_full = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    sky_image_full.setTo(cv::Scalar(0, 0, 255), sky_mask);
    cv::addWeighted(_src_img, 1, sky_image_full, 1, 0, sky_image);

    cv::imwrite(output_path, sky_image);

    LOG(INFO) << "图像: " << image_file_path << "检测完毕";
}

void SkyAreaDetector::batch_detect(const std::string &image_dir, const std::string &output_dir) {

    // 获取图像信息
    std::vector<std::string> image_file_list;
    file_processor::FileSystemProcessor::get_directory_files(image_dir,
            image_file_list,
            ".jpg",
            file_processor::FileSystemProcessor::
            SEARCH_OPTION_T::ALLDIRECTORIES);

    LOG(INFO) << "开始批量提取天空区域";
    LOG(INFO) << "--- 图像: --- 耗时(s): ---";

    for (auto &image_file : image_file_list) {

        auto start_t = std::chrono::high_resolution_clock::now();

        auto image_file_name = file_processor::FileSystemProcessor::get_file_name(image_file);
        auto output_path = file_processor::FileSystemProcessor::combine_path(output_dir, image_file_name);

        // 加载图像
        load_image(image_file);

        // 提取天空区域
        cv::Mat sky_mask;

        if (extract_sky(_src_img, sky_mask)) {

            // 制作掩码输出
            cv::Mat sky_image;

            int image_height = _src_img.size[0];
            int image_width = _src_img.size[1];

            cv::Mat sky_image_full = cv::Mat::zeros(image_height, image_width, CV_8UC3);
            _src_img.setTo(cv::Scalar(0, 0, 255), sky_mask);
//                cv::addWeighted(_src_img, 1, sky_image_full, 1, 0, sky_image);

            cv::imwrite(output_path, _src_img);

            auto end_t = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> cost_time = end_t - start_t;

            LOG(INFO) << "---- " << image_file_name << " ---- "
                      << cost_time.count() << "s" << std::endl;
        } else {

            cv::imwrite(output_path, _src_img);

            LOG(INFO) << "---- " << image_file_name << " ---- "
                      << "Null s" << std::endl;
        }
    }

    LOG(INFO) << "批量提取完毕";
}

/***
 * 提取图像梯度信息
 * @param src_image
 * @param gradient_image
 */
void SkyAreaDetector::extract_image_gradient(const cv::Mat &src_image, cv::Mat &gradient_image) {
    // 转灰度图像
    cv::Mat gray_image;
    cv::cvtColor(src_image, gray_image, cv::COLOR_BGR2GRAY);
    // Sobel算子提取图像梯度信息
    cv::Mat x_gradient;
    cv::Sobel(gray_image, x_gradient, CV_64F, 1, 0);
    cv::Mat y_gradient;
    cv::Sobel(gray_image, y_gradient, CV_64F, 0, 1);
    // 计算梯度信息图
    cv::Mat gradient;
    cv::pow(x_gradient, 2, x_gradient);
    cv::pow(y_gradient, 2, y_gradient);
    cv::add(x_gradient, y_gradient, gradient);
    cv::sqrt(gradient, gradient);

    gradient_image = gradient;

}

/***
 * 计算天空边界线
 * @param src_image
 * @return
 */
std::vector<int> SkyAreaDetector::extract_border_optimal(const cv::Mat &src_image) {
    // 提取梯度信息图
    cv::Mat gradient_info_map;
    extract_image_gradient(src_image, gradient_info_map);

    int n = static_cast<int>(std::floor((f_thres_sky_max - f_thres_sky_min)
                                        / f_thres_sky_search_step)) + 1;

    std::vector<int> border_opt;
    double jn_max = 0.0;

    for (int k = 1; k < n + 1; k++) {
        double t = f_thres_sky_min + (std::floor((f_thres_sky_max - f_thres_sky_min) / n) - 1) * (k - 1);

        std::vector<int> b_tmp = extract_border(gradient_info_map, t);
        double jn = calculate_sky_energy(b_tmp, src_image);
        if (std::isinf(jn)) {
            LOG(INFO) << "Jn is -inf" << std::endl;
        }

        if (jn > jn_max) {
            jn_max = jn;
            border_opt = b_tmp;
        }
    }

    return border_opt;
}

/***
 * 计算天空边界线
 * @param gradient_info_map
 * @param thresh
 * @return
 */
std::vector<int> SkyAreaDetector::extract_border(const cv::Mat &gradient_info_map,
        double thresh) {
    int image_height = gradient_info_map.size[0];
    int image_width = gradient_info_map.size[1];
    std::vector<int> border(image_width, image_height - 1);

    for (int col = 0; col < image_width; ++col) {
        int row_index = 0;
        for (int row = 0; row < image_height; ++row) {
            row_index = row;
            if (gradient_info_map.at<double>(row, col) > thresh) {
                border[col] = row;
                break;
            }
        }
        if (row_index == 0) {
            border[col] = image_height - 1;
        }
    }

    return border;
}

/***
 * 改善天空边界线
 * @param border
 * @param src_image
 * @return
 */
std::vector<int> SkyAreaDetector::refine_border(const std::vector<int> &border,
        const cv::Mat &src_image) {

    int image_height = src_image.size[0];
    int image_width = src_image.size[1];

    // 制作天空图像掩码和地面图像掩码
    cv::Mat sky_mask = make_sky_mask(src_image, border, 1);
    cv::Mat ground_mask = make_sky_mask(src_image, border, 0);

    // 扣取天空图像和地面图像
    cv::Mat sky_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    cv::Mat ground_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    src_image.copyTo(sky_image, sky_mask);
    src_image.copyTo(ground_image, ground_mask);

    // 计算天空和地面图像协方差矩阵
    int ground_non_zeros_nums = cv::countNonZero(ground_mask);
    int sky_non_zeros_nums = cv::countNonZero(sky_mask);

    cv::Mat ground_image_non_zero = cv::Mat::zeros(ground_non_zeros_nums, 3, CV_8UC1);
    cv::Mat sky_image_non_zero = cv::Mat::zeros(sky_non_zeros_nums, 3, CV_8UC1);

    int row_index = 0;
    for (int col = 0; col < ground_image.cols; ++col) {
        for (int row = 0; row < ground_image.rows; ++row) {
            if (ground_image.at<cv::Vec3b>(row, col)[0] == 0 &&
                    ground_image.at<cv::Vec3b>(row, col)[1] == 0 &&
                    ground_image.at<cv::Vec3b>(row, col)[2] == 0) {
                continue;
            } else {
                cv::Vec3b intensity = ground_image.at<cv::Vec3b>(row, col);
                ground_image_non_zero.at<uchar>(row_index, 0) = intensity[0];
                ground_image_non_zero.at<uchar>(row_index, 1) = intensity[1];
                ground_image_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    row_index = 0;
    for (int col = 0; col < sky_image.cols; ++col) {
        for (int row = 0; row < sky_image.rows; ++row) {
            if (sky_image.at<cv::Vec3b>(row, col)[0] == 0 &&
                    sky_image.at<cv::Vec3b>(row, col)[1] == 0 &&
                    sky_image.at<cv::Vec3b>(row, col)[2] == 0) {
                continue;
            } else {
                cv::Vec3b intensity = sky_image.at<cv::Vec3b>(row, col);
                sky_image_non_zero.at<uchar>(row_index, 0) = intensity[0];
                sky_image_non_zero.at<uchar>(row_index, 1) = intensity[1];
                sky_image_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    // k均值聚类调整天空区域边界
    cv::Mat sky_image_float;
    sky_image_non_zero.convertTo(sky_image_float, CV_32FC1);
    cv::Mat labels;
    cv::kmeans(sky_image_float, 2, labels,
               cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
               10, cv::KMEANS_RANDOM_CENTERS);
    int label_1_nums = cv::countNonZero(labels);
    int label_0_nums = labels.rows - label_1_nums;

    cv::Mat sky_label_1_image = cv::Mat::zeros(label_1_nums, 3, CV_8UC1);
    cv::Mat sky_label_0_image = cv::Mat::zeros(label_0_nums, 3, CV_8UC1);

    row_index = 0;
    for (int row = 0; row < labels.rows; ++row) {
        if (labels.at<float>(row, 0) == 0.0) {
            sky_label_0_image.at<uchar>(row_index, 0) = sky_image_non_zero.at<uchar>(row, 0);
            sky_label_0_image.at<uchar>(row_index, 1) = sky_image_non_zero.at<uchar>(row, 1);
            sky_label_0_image.at<uchar>(row_index, 2) = sky_image_non_zero.at<uchar>(row, 2);
            row_index++;
        }
    }
    row_index = 0;
    for (int row = 0; row < labels.rows; ++row) {
        if (labels.at<float>(row, 0) == 1.0) {
            sky_label_1_image.at<uchar>(row_index, 0) = sky_image_non_zero.at<uchar>(row, 0);
            sky_label_1_image.at<uchar>(row_index, 1) = sky_image_non_zero.at<uchar>(row, 1);
            sky_label_1_image.at<uchar>(row_index, 2) = sky_image_non_zero.at<uchar>(row, 2);
            row_index++;
        }
    }

    cv::Mat sky_covar_1;
    cv::Mat sky_mean_1;
    cv::calcCovarMatrix(sky_label_1_image, sky_covar_1,
                        sky_mean_1, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
    cv::Mat ic_s1;
    cv::invert(sky_covar_1, ic_s1, cv::DECOMP_SVD);

    cv::Mat sky_covar_0;
    cv::Mat sky_mean_0;
    cv::calcCovarMatrix(sky_label_0_image, sky_covar_0,
                        sky_mean_0, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
    cv::Mat ic_s0;
    cv::invert(sky_covar_0, ic_s0, cv::DECOMP_SVD);

    cv::Mat ground_covar;
    cv::Mat ground_mean;
    cv::calcCovarMatrix(ground_image_non_zero, ground_covar,
                        ground_mean, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
    cv::Mat ic_g;
    cv::invert(ground_covar, ic_g, cv::DECOMP_SVD);

    cv::Mat sky_mean;
    cv::Mat sky_covar;
    cv::Mat ic_s;
    if (cv::Mahalanobis(sky_mean_0, ground_mean, ic_s0) > cv::Mahalanobis(sky_mean_1, ground_mean, ic_s1)) {
        sky_mean = sky_mean_0;
        sky_covar = sky_covar_0;
        ic_s = ic_s0;
    } else {
        sky_mean = sky_mean_1;
        sky_covar = sky_covar_1;
        ic_s = ic_s1;
    }

    std::vector<int> border_new(border.size(), 0);
    for (size_t col = 0; col < border.size(); ++col) {
        double cnt = 0.0;
        for (int row = 0; row < border[col]; ++row) {
            // 计算原始天空区域的区域每个像素点和修正过后的天空区域的每个点的Mahalanobis距离
            cv::Mat ori_pix;
            src_image.row(row).col(static_cast<int>(col)).convertTo(ori_pix, sky_mean.type());
            ori_pix = ori_pix.reshape(1, 1);
            double distance_s = cv::Mahalanobis(ori_pix,
                                                sky_mean, ic_s);
            double distance_g = cv::Mahalanobis(ori_pix,
                                                ground_mean, ic_g);

            if (distance_s < distance_g) {
                cnt++;
            }
        }
        if (cnt < (border[col] / 2)) {
            border_new[col] = 0;
        } else {
            border_new[col] = border[col];
        }
    }

    return border_new;
}

/***
 * 计算天空图像能量函数
 * @param border
 * @param src_image
 * @return
 */
double SkyAreaDetector::calculate_sky_energy(const std::vector<int> &border,
        const cv::Mat &src_image) {

    int image_height = src_image.size[0];
    int image_width = src_image.size[1];

    // 制作天空图像掩码和地面图像掩码
    cv::Mat sky_mask = make_sky_mask(src_image, border, 1);
    cv::Mat ground_mask = make_sky_mask(src_image, border, 0);

    // 扣取天空图像和地面图像
    cv::Mat sky_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    cv::Mat ground_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    src_image.copyTo(sky_image, sky_mask);
    src_image.copyTo(ground_image, ground_mask);

    // 计算天空和地面图像协方差矩阵
    int ground_non_zeros_nums = cv::countNonZero(ground_mask);
    int sky_non_zeros_nums = cv::countNonZero(sky_mask);

    if (ground_non_zeros_nums == 0 || sky_non_zeros_nums == 0) {
        return std::numeric_limits<double>::min();
    }

    cv::Mat ground_image_non_zero = cv::Mat::zeros(ground_non_zeros_nums, 3, CV_8UC1);
    cv::Mat sky_image_non_zero = cv::Mat::zeros(sky_non_zeros_nums, 3, CV_8UC1);

    int row_index = 0;
    for (int col = 0; col < ground_image.cols; ++col) {
        for (int row = 0; row < ground_image.rows; ++row) {
            if (ground_image.at<cv::Vec3b>(row, col)[0] == 0 &&
                    ground_image.at<cv::Vec3b>(row, col)[1] == 0 &&
                    ground_image.at<cv::Vec3b>(row, col)[2] == 0) {
                continue;
            } else {
                cv::Vec3b intensity = ground_image.at<cv::Vec3b>(row, col);
                ground_image_non_zero.at<uchar>(row_index, 0) = intensity[0];
                ground_image_non_zero.at<uchar>(row_index, 1) = intensity[1];
                ground_image_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    row_index = 0;
    for (int col = 0; col < sky_image.cols; ++col) {
        for (int row = 0; row < sky_image.rows; ++row) {
            if (sky_image.at<cv::Vec3b>(row, col)[0] == 0 &&
                    sky_image.at<cv::Vec3b>(row, col)[1] == 0 &&
                    sky_image.at<cv::Vec3b>(row, col)[2] == 0) {
                continue;
            } else {
                cv::Vec3b intensity = sky_image.at<cv::Vec3b>(row, col);
                sky_image_non_zero.at<uchar>(row_index, 0) = intensity[0];
                sky_image_non_zero.at<uchar>(row_index, 1) = intensity[1];
                sky_image_non_zero.at<uchar>(row_index, 2) = intensity[2];
                row_index++;
            }
        }
    }

    cv::Mat ground_covar;
    cv::Mat ground_mean;
    cv::Mat ground_eig_vec;
    cv::Mat ground_eig_val;
    cv::calcCovarMatrix(ground_image_non_zero, ground_covar,
                        ground_mean, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
    cv::eigen(ground_covar, ground_eig_val, ground_eig_vec);

    cv::Mat sky_covar;
    cv::Mat sky_mean;
    cv::Mat sky_eig_vec;
    cv::Mat sky_eig_val;
    cv::calcCovarMatrix(sky_image_non_zero, sky_covar,
                        sky_mean, CV_COVAR_ROWS | CV_COVAR_SCALE | CV_COVAR_NORMAL);
    cv::eigen(sky_covar, sky_eig_val, sky_eig_vec);

    int para = 2; // 论文原始参数
    double ground_det = cv::determinant(ground_covar);
    double sky_det = cv::determinant(sky_covar);
    double ground_eig_det = cv::determinant(ground_eig_vec);
    double sky_eig_det = cv::determinant(sky_eig_vec);

    return 1 / ((para * sky_det + ground_det) + (para * sky_eig_det + ground_eig_det));

}

/***
 * 确定图像是否含有天空区域
 * @param border
 * @param thresh_1
 * @param thresh_2
 * @param thresh_3
 * @return
 */
bool SkyAreaDetector::has_sky_region(const std::vector<int> &border,
                                     double thresh_1, double thresh_2,
                                     double thresh_3) {
    double border_mean = 0.0;
    for (size_t i = 0; i < border.size(); ++i) {
        border_mean += border[i];
    }
    border_mean /= border.size();

    // 如果平均天际线太小认为没有天空区域
    if (border_mean < thresh_1) {
        return false;
    }

    std::vector<int> border_diff(static_cast<int>(border.size() - 1), 0);
    for (auto i = static_cast<int>(border.size() - 1); i >= 0; i--) {
        border_diff[i] = std::abs(border[i + 1] - border[i]);
    }
    double border_diff_mean = 0.0;
    for (auto &diff_val : border_diff) {
        border_diff_mean += diff_val;
    }
    border_diff_mean /= border_diff.size();

    return !(border_mean < thresh_1 || (border_diff_mean > thresh_3 && border_mean < thresh_2));
}

/***
 * 判断图像是否有部分区域为天空区域
 * @param border
 * @param thresh_1
 * @return
 */
bool SkyAreaDetector::has_partial_sky_region(const std::vector<int> &border,
        double thresh_1) {
    std::vector<int> border_diff(static_cast<int>(border.size() - 1), 0);
    for (int i = static_cast<int>(border.size() - 1); i >= 0; i--) {
        border_diff[i] = std::abs(border[i + 1] - border[i]);
    }

    for (size_t i = 0; i < border_diff.size(); ++i) {
        if (border_diff[i] > thresh_1) {
            return true;
        }
    }

    return false;
}

/***
 * 天空区域和原始图像融合图
 * @param src_image
 * @param border
 * @param sky_image
 */
void SkyAreaDetector::display_sky_region(const cv::Mat &src_image,
        const std::vector<int> &border,
        cv::Mat &sky_image) {

    int image_height = src_image.size[0];
    int image_width = src_image.size[1];
    // 制作天空图掩码
    cv::Mat sky_mask = make_sky_mask(src_image, border, 1);

    // 天空和原始图像融合
    cv::Mat sky_image_full = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    sky_image_full.setTo(cv::Scalar(0, 0, 255), sky_mask);
    cv::addWeighted(src_image, 1, sky_image_full, 1, 0, sky_image);
}

/***
 * 制作天空掩码图像
 * @param src_image
 * @param border
 * @param type: 1: 天空 0: 地面
 * @return
 */
cv::Mat SkyAreaDetector::make_sky_mask(const cv::Mat &src_image,
                                       const std::vector<int> &border,
                                       int type) {
    int image_height = src_image.size[0];
    int image_width = src_image.size[1];

    cv::Mat mask = cv::Mat::zeros(image_height, image_width, CV_8UC1);

    if (type == 1) {
        for (int col = 0; col < image_width; ++col) {
            for (int row = 0; row < image_height; ++row) {
                if (row <= border[col]) {
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }
    } else if (type == 0) {
        for (int col = 0; col < image_width; ++col) {
            for (int row = 0; row < image_height; ++row) {
                if (row > border[col]) {
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }
    } else {
        assert(type == 0 || type == 1);
    }

    return mask;
}

}

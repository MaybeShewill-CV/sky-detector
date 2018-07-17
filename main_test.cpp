// Copyright 2014 Baidu Inc. All Rights Reserved.
// Author: Luo Yao (luoyao@baidu.com)
// File: main_test.cpp
// Date: 18-6-29 下午2:24

#include <glog/logging.h>

#include <imageSkyDetector.h>
#include <file_system_processor.h>

#define BATCH_PROCESS
//#define SINGLE_PROCESS


int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetLogDestination(google::GLOG_INFO, "./log/image_quality_check_");
    google::SetStderrLogging(google::GLOG_INFO);

    // 创建log文件夹
    if (!file_processor::FileSystemProcessor::is_directory_exist("./log")) {
        file_processor::FileSystemProcessor::create_directories("./log");
    }

    sky_detector::SkyAreaDetector detector;

    if (argc != 3) {
        LOG(INFO) << "Usage:";
        LOG(INFO) << "./detector 图像输入路径 图像输出路径";
        return -1;
    } else {

#ifdef BATCH_PROCESS
        detector.batch_detect(argv[1], argv[2]);
#endif

#ifdef SINGLE_PROCESS
        detector.detect(argv[1], argv[2]);
#endif
        return 1;
    }
}
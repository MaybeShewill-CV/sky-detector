// Copyright 2014 Baidu Inc. All Rights Reserved.
// Author: Luo Yao (luoyao@baidu.com)
// File: main_test.cpp
// Date: 18-6-29 下午2:24

#include <glog/logging.h>

#include <imageSkyDetector.h>
#include <file_system_processor.h>


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
    detector.detect(argv[1], argv[2]);

}
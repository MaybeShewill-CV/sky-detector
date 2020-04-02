cmake_minimum_required(VERSION 3.5.1)
project(sky-detector)

set(CMAKE_CXX_STANDARD 11)

set(MAIN_SOURCE_FILE main_test.cpp)
aux_source_directory(sky_detector DIR_SRCS)
aux_source_directory(file_processor DIR_SRCS)
aux_source_directory(base DIR_SRCS)

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -Wall")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Boost REQUIRED)
if(!Boost_FOUND)
    message("Boost NOT FOUND")
endif()

find_package( OpenCV REQUIRED )
if(!Opencv_FOUND)
    message("OpenCV NOT FOUND")
endif()

include_directories (
        /usr/include
        /usr/local/include
        ./sky_detector
        ./file_processor
        ./base
        ${OpenMP_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(detector ${MAIN_SOURCE_FILE} ${DIR_SRCS})

target_link_libraries(detector
        ${OpenCV_LIBS}
        /usr/local/lib/libboost_filesystem.so
        /usr/local/lib/libboost_system.so
        /usr/local/lib/libglog.so
        )
LINK_DIRECTORIES(/usr/local/lib /usr/lib/x86_64-linux-gnu)

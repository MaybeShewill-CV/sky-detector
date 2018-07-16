cmake_minimum_required(VERSION 3.9)
project(sky-detector)

set(CMAKE_CXX_STANDARD 11)

set(MAIN_SOURCE_FILE main_test.cpp)
aux_source_directory(sky_detector DIR_SRCS)
aux_source_directory(file_processor DIR_SRCS)


set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

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
        ${OpenMP_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${Opencv_INCLUDE_DIRS}
)

set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
add_executable(detector ${MAIN_SOURCE_FILE} ${DIR_SRCS})

target_link_libraries(detector -static
        ${OpenCV_LIBS}
        /usr/local/lib/libboost_filesystem.a
        /usr/local/lib/libboost_system.a
        /usr/local/lib/libglog.a
        )
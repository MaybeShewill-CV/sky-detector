# Sky-Area-Detector
sky area detection without deep neural networks.
A c++ implemention of the algorithm mentioned in paper 
"Sky Region Detection in a Single Image for Autonomous Ground
Robot Navigation". A fast and robust method to extract the 
sky area of an image.
  
# Installation
This implementation need opencv library. Since the project is 
static compiled for conveniently transplanting you need static
opencv library linked to this project. If static compilation is
not necessary for you feel free to modify the CMakeList to dynamic
compile the project.

This software has only been tested on ubuntu 16.04(x64), opencv3.4. 
To install this package your compiler need to support C++11. 

```
git clone https://github.com/MaybeShewill-CV/sky-detector.git
```

# Build

```
cd ROOT_FOLDER_DIR
mkdir build
cd build
cmake ..
make -j
```

The project will generate a static binary file which can be used
on other platform without any dynamic library. The binary file 
built on Ubuntu 16.04LTS was tested on CentOS 6 and worked 
correctly.

# Usage

```
cd build_dir
./detect input_image_file_path output_image_file_path
```

#### 结果示意图如下

`Test input image with full sky`

![Test_input_full_sky](/data/full_sky.png)

`Test input image with ful sky result`

![Test_input_full_sky_result](/data/ret_mask.jpg)

`Test input image with part sky`

![Test_input_part_sky](/data/partial_sky.png)

`Test input image with part sky result`

![Test_input_part_sky_result](/data/ret2.jpg)

# TODO
- [ ] Accelerate the calculation process

FREAK: Fast Retina Keypoint

Alexandre Alahi, Raphael Ortiz, Kirell Benzi, Pierre Vandergheynst
Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland

http://www.ivpe.com/freak.htm
http://infoscience.epfl.ch/record/175537/files/2069.pdf
http://lts2.epfl.ch

---------------
DEPENDENCIES

CMake >= 2.6, http://www.cmake.org/
OpenCV >= 2.4 , http://code.opencv.org/projects/opencv

---------------
BUILDING

### LIBRARY + DEMO

Create a build directory
mkdir build
cd build

- SET CMAKE_INSTALL_PREFIX (default /usr/local/)
- SET OpenCV_DIR if not installed in default dir

cmake -DCMAKE_INSTALL_PREFIX=/Users/YourName/path/toFolder/ -DOpenCV_DIR=/Users/YourName/path/to/OpenCv/ ..

### NO DEMO

- SET BUILD_DEMO=OFF

cmake -DCMAKE_INSTALL_PREFIX=/Users/YourName/path/toFolder/ -DOpenCV_DIR=/Users/YourName/path/to/OpenCv/ -DBUILD_DEMO=OFF ..

make
(sudo) make install






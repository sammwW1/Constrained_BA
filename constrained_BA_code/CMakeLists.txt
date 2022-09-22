
cmake_minimum_required( VERSION 2.8 )
project( g2o_ba_example )

set( CMAKE_BUILD_TYPE Release )
# set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )
# set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CXX_STANDARD 14)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( Python3 COMPONENTS Intrepreter Development Numpy )
find_package( G2O REQUIRED )
#find_package( OpenCV REQUIRED )
find_package(OpenCV 4.6.0 REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Cholmod )
find_package(Ceres REQUIRED)
find_package(matplotlib_cpp)

include_directories( ${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR} ${matplotlib_SOURCE_DIR} ${Python3_INCLUDE_DIRS} include)



include_directories(SYSTEM)



add_executable( ba_example main.cpp )
target_link_libraries( ba_example 
    ${OpenCV_LIBS}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
    ${CHOLMOD_LIBRARIES}
    ${Python3_LIBRARIES}
    )



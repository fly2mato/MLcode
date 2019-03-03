cmake_minimum_required(VERSION 3.4)

project(mlcode)

#find_package(Eigen3 REQUIRED)
#Bring the headers, such as Student.h into the project
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++11 -O3")
endif(NOT WIN32)

#Can manually add the sources using the set command as follows:
#set(SOURCES src/mainapp.cpp src/Student.cpp)
#set(CMAKE_CXX_COMPILER      "clang++" )         # 显示指定使用的C++编译器
#set(CMAKE_CXX_FLAGS   "-std=c++11")             # c++11
set(CMAKE_CXX_FLAGS   "-g")                     # 调试信息
set(CMAKE_CXX_FLAGS   "-Wall")                  # 开启所有警告

set(CMAKE_CXX_FLAGS_DEBUG   "-O0" )             # 调试包不优化
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG " )   # release包优化

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR})
#However, the file(GLOB...) allows for wildcard additions:

#add_executable(findpath ${SOURCES})
#add_executable(mytest "mytest/mytest.cpp" "src/LinearRegression.cpp" "src/LogisticRegression.cpp")
add_executable(DTtest "mytest/DTtest.cpp" "src/DecisionTree.cpp")

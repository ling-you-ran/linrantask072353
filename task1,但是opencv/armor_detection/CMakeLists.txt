cmake_minimum_required(VERSION 3.10)
project(armor_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(armor_detection ../main.cpp)

target_link_libraries(armor_detection ${OpenCV_LIBS})

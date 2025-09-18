cmake_minimum_required(VERSION 3.10)
project(ArmorDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV first.")
endif()

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation directory")

if(ONNXRUNTIME_DIR STREQUAL "")
    message(FATAL_ERROR "Please specify ONNX Runtime directory using -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIRS onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_DIR}/include)
if(WIN32)
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIB})
else()
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIB})
endif()

if(ONNXRUNTIME_INCLUDE_DIRS AND ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_DIR}")
    include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "ONNX Runtime not found in ${ONNXRUNTIME_DIR}")
endif()

add_executable(armor_detection armor_detection_enhanced.cpp)

target_link_libraries(armor_detection 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(armor_detection pthread)
endif()

install(TARGETS armor_detection DESTINATION bin)

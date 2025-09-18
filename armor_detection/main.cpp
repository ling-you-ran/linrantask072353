#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include <chrono>

enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };
enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };

namespace fs = std::filesystem;

ArmorColor getColor(float* output_data) {
    int max_idx = 9;
    for (int i = 10; i <= 12; ++i) {
        if (output_data[i] > output_data[max_idx]) {
            max_idx = i;
        }
    }
    return static_cast<ArmorColor>(max_idx - 9);
}

ArmorNumber getNumber(float* output_data) {
    int max_idx = 13;
    for (int i = 14; i <= 20; ++i) {
        if (output_data[i] > output_data[max_idx]) {
            max_idx = i;
        }
    }
    int num_idx = max_idx - 13;
    return (num_idx >= 0 && num_idx <= 7) ? static_cast<ArmorNumber>(num_idx) : ArmorNumber::UNKNOWN;
}

std::string convertNumberToString(ArmorNumber number) {
    switch (number) {
        case ArmorNumber::SENTRY: return "SENTRY";
        case ArmorNumber::NO1: return "NO1";
        case ArmorNumber::NO2: return "NO2";
        case ArmorNumber::NO3: return "NO3";
        case ArmorNumber::NO4: return "NO4";
        case ArmorNumber::NO5: return "NO5";
        case ArmorNumber::OUTPOST: return "OUTPOST";
        case ArmorNumber::BASE: return "BASE";
        default: return "UNKNOWN";
    }
}

int main() {
    fs::path buildPath = fs::current_path();
    fs::path projectPath = buildPath.parent_path();
    std::string modelPath = (projectPath / "model" / "opt-1208-001.onnx").string();
    std::string videoPath = (projectPath / "video" / "jiao.avi").string();
    

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ArmorDetection");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  // 启用基本优化

    try {
        Ort::Session session(env, modelPath.c_str(), session_options);
        std::cout << "Model loaded successfully!" << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputName(0, allocator);
        auto output_name = session.GetOutputName(0, allocator);
        std::vector<const char*> output_names = {output_name};
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << videoPath << std::endl;
            return -1;
        }

        cv::namedWindow("Armor Detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Armor Detection", 800, 600);
        
        int frame_count = 0;
        auto total_start = std::chrono::high_resolution_clock::now();
        
        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            auto frame_start = std::chrono::high_resolution_clock::now();
            
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(416, 416));
            
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
            
            resized.convertTo(resized, CV_32F);
            
            std::vector<float> input_tensor_values(3 * 416 * 416);
            int index = 0;
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 416; ++h) {
                    for (int w = 0; w < 416; ++w) {
                        input_tensor_values[index++] = resized.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            std::vector<int64_t> input_shape = {1, 3, 416, 416};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                3 * 416 * 416,
                input_shape.data(),
                4
            );

            std::vector<Ort::Value> output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                &input_name,
                &input_tensor,
                1,
                output_names.data(),
                output_names.size()
            );

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            float confidence = output_data[8];
            
            if (confidence > 0.5) {
                std::vector<cv::Point2f> corners;
                for (int i = 0; i < 8; i += 2) {
                    float x = output_data[i] * frame.cols / 416.0f;
                    float y = output_data[i+1] * frame.rows / 416.0f;
                    corners.emplace_back(x, y);
                }
                
                cv::Scalar color;
                ArmorColor armor_color = getColor(output_data);
                switch (armor_color) {
                    case ArmorColor::BLUE: color = cv::Scalar(255, 0, 0); break;
                    case ArmorColor::RED: color = cv::Scalar(0, 0, 255); break;
                    case ArmorColor::PURPLE: color = cv::Scalar(255, 0, 255); break;
                    default: color = cv::Scalar(128, 128, 128);
                }
                

                cv::line(frame, corners[0], corners[2], color, 2);  
                cv::line(frame, corners[1], corners[3], color, 2);  
                
                cv::line(frame, corners[0], corners[1], color, 2);  
                cv::line(frame, corners[3], corners[2], color, 2);  
                
                ArmorNumber armor_number = getNumber(output_data);
                std::string number_str = convertNumberToString(armor_number);
                
                cv::putText(frame, number_str, cv::Point(10, frame.rows - 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
                
                std::string info = "Conf: " + std::to_string(confidence).substr(0, 4);
                cv::putText(frame, info, corners[0] + cv::Point2f(0, -10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
            
            frame_count++;
            auto frame_end = std::chrono::high_resolution_clock::now();
            double fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(
                frame_end - frame_start).count();
            
            std::string fps_text = "FPS: " + std::to_string((int)fps);
            cv::putText(frame, fps_text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);


            cv::imshow("Armor Detection", frame);
            
            if (cv::waitKey(1) == 27) break;
        }
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::seconds>(
            total_end - total_start).count();
        std::cout << "Total frames processed: " << frame_count << std::endl;
        std::cout << "Average FPS: " << frame_count / total_time << std::endl;

        cap.release();
        cv::destroyAllWindows();
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

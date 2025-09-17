#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <iostream>
#include <filesystem>

enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };
enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };

namespace fs = std::filesystem;

int main() {
    fs::path buildPath = fs::current_path();
    fs::path projectPath = buildPath.parent_path();
    
    std::string modelPath = (projectPath / "model" / "opt-1208-001.onnx").string();
    std::string videoPath = (projectPath / "video" / "jiao.avi").string();
    
    std::cout << "Build directory: " << buildPath << std::endl;
    std::cout << "Project directory: " << projectPath << std::endl;
    std::cout << "Model path: " << modelPath << std::endl;
    std::cout << "Video path: " << videoPath << std::endl;

    if (!fs::exists(modelPath)) {
        std::cerr << "Error: Model file not found at: " << modelPath << std::endl;
        std::cerr << "Please make sure the file exists in the model/ folder" << std::endl;
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    std::cout << "Model loaded successfully!" << std::endl;
    
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera, trying video file..." << std::endl;
        
        if (fs::exists(videoPath)) {
            cap.open(videoPath);
            if (!cap.isOpened()) {
                std::cerr << "Cannot open video file either" << std::endl;
                return -1;
            }
            std::cout << "Video file opened successfully!" << std::endl;
        } else {
            std::cerr << "Video file not found: " << videoPath << std::endl;
            return -1;
        }
    } else {
        std::cout << "Camera opened successfully!" << std::endl;
    }

    std::cout << "Press ESC to exit" << std::endl;

    cv::namedWindow("Armor Detection", cv::WINDOW_NORMAL);

    int frameCount = 0;
    double totalTime = 0.0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Frame is empty or video ended" << std::endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat input;
        cv::resize(frame, input, cv::Size(416, 416));
        net.setInput(input);
        cv::Mat output = net.forward();
        const float* data = (float*)output.data;
        int numDetections = output.size[1];
        
        float scaleX = (float)frame.cols / 416;
        float scaleY = (float)frame.rows / 416;

        for (int i = 0; i < numDetections; ++i) {
            float confidence = data[i * 21 + 8];
            if (confidence < 0.5f) continue;

            std::vector<cv::Point> points;
            for (int j = 0; j < 4; ++j) {
                int x = data[i * 21 + j * 2] * scaleX;
                int y = data[i * 21 + j * 2 + 1] * scaleY;
                points.push_back(cv::Point(x, y));
            }

            void draw_armor(cv::Mat& image, const ArmorDetection& armor, int detectedNumber) {
    cv::line(image, armor.corners[0], armor.corners[2], cv::Scalar(0, 255, 0), 2);  
    cv::line(image, armor.corners[1], armor.corners[3], cv::Scalar(0, 255, 0), 2);  
    cv::line(image, armor.corners[0], armor.corners[3], cv::Scalar(0, 0, 255), 2);  
    cv::line(image, armor.corners[1], armor.corners[2], cv::Scalar(0, 0, 255), 2);  
    std::string text = "NO" + std::to_string(detectedNumber);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalTime += inferenceTime;
        frameCount++;

        double fps = 1000.0 / inferenceTime;
        std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 4);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Armor Detection", frame);
        if (cv::waitKey(1) == 27) break;
    }

    if (frameCount > 0) {
        double avgTime = totalTime / frameCount;
        double avgFPS = 1000.0 / avgTime;
        std::cout << "Average inference time: " << avgTime << "ms" << std::endl;
        std::cout << "Average FPS: " << avgFPS << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

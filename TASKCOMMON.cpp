#include <iostream>
#include <thread>
#include <chrono>

const int MAX_TASKS = 1000;
const int MAX_PROCESSORS = 4;
const int PUBLISH_DURATION = 10;

// 参数类
class ProcessParam {
private:
    bool processed_ = false;
public:
    void markProcessed() { processed_ = true; }
    bool isProcessed() const { return processed_; }
};

// 模拟处理器
class Processor {
private:
    int* internal_data_;
    int id_;
public:
    Processor(int id = 0) : id_(id) {
        internal_data_ = new int(0);
    }
    Processor(const Processor& other) : id_(other.id_) {
        std::cout << "Processor " << id_ << " copy started (10s)...\n";
        std::this_thread::sleep_for(std::chrono::seconds(10));
        internal_data_ = other.internal_data_; 
        std::cout << "Processor " << id_ << " copy finished.\n";
    }
    ~Processor() { delete internal_data_; }

    void process(ProcessParam& param) {
        int* temp = internal_data_;
        internal_data_ = nullptr;
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 模拟处理耗时
        *temp += 1;
        param.markProcessed();
        internal_data_ = temp;
    }

    int getId() const { return id_; }
};
//-----------------------------------------------------------------
ProcessParam* task_buffer[MAX_TASKS];
int task_count = 0;
bool is_publishing = true;
int processed_count = 0;
Processor processors[MAX_PROCESSORS];
// 单个处理器处理数据
void push_task(ProcessParam* param) {
    while (task_count >= MAX_TASKS) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    task_buffer[task_count++] = param;
}
// 统计功能
ProcessParam* pop_task() {
    if (task_count <= 0) return nullptr;
    ProcessParam* param = task_buffer[0];
    for (int i = 0; i < task_count - 1; ++i) {
        task_buffer[i] = task_buffer[i + 1];
    }
    task_count--;
    return param;
}

// 处理功能
void processor_worker(int processor_id) {
    Processor& processor = processors[processor_id];
    
    while (true) {
        if (!is_publishing && task_count == 0) break;
        
        ProcessParam* param = pop_task();
        if (param) {
            processor.process(*param);
            processed_count++;
            delete param;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::cout << "Processor " << processor_id << " finished\n";
}
//初始数据
void publisher() {
    auto end_time = std::chrono::steady_clock::now() + 
                   std::chrono::seconds(PUBLISH_DURATION);
    
    while (std::chrono::steady_clock::now() < end_time) {
        ProcessParam* param = new ProcessParam();
        push_task(param);
    }
    
    is_publishing = false;
    std::cout << "Publisher finished\n";
}
int main() {
    for (int i = 0; i < MAX_PROCESSORS; ++i) {
        processors[i] = Processor(i);
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // 处理器并行
    std::thread processor_threads[MAX_PROCESSORS];
    for (int i = 0; i < MAX_PROCESSORS; ++i) {
        processor_threads[i] = std::thread(processor_worker, i);
    }
    
    // 数据初始化
    std::thread pub_thread(publisher);
    pub_thread.join();
    for (int i = 0; i < MAX_PROCESSORS; ++i) {
        processor_threads[i].join();
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double fps = processed_count / elapsed.count();
//-----------------------------------------------------------------------------
    std::cout << "\n结束\n";
    std::cout << "总处理数: " << processed_count << "\n";
    std::cout << "耗时: " << elapsed.count() << "秒\n";
    std::cout << "fps: " << fps << "\n";
    
    return 0;
}

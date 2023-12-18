#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream> 
#include "util.hpp"

#define KERNEL_FILE "rotate_kernel.cl"
 

//------------------------------------------------------------------------------
//
//  Function to analyze and output results
//
//------------------------------------------------------------------------------
void results( double run_time)
{

    float mflops = 0.0f;
    
    //mflops = 2.0 * N * N * N/(1000000.0f * run_time);
    printf(" %.5f seconds at %.1f MFLOPS \n",  run_time,mflops);

}


int main() {

      std::cout << "OpenCV version: " << CV_VERSION << std::endl;
       #ifdef WITH_OPENCL
        std::cout << "OpenCL is enabled in this build." << std::endl;
    #else
        std::cout << "OpenCL is not enabled in this build." << std::endl;
    #endif
    
    // Load the image 
    cv::Mat image = cv::imread("dog.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat image = cv::imread("./dog.jpg", cv::IMREAD_COLOR);

     std::cout << "image matrix size: \n rows: " << image.rows <<", cols: "<< image.cols <<"; type: "<< image.type() <<", "<<  std::endl;
    std::cout << "image matrix dimension : " << image.dims<< std::endl; std::cout << "image total : " << image.total() << std::endl; 
    std::cout << "image elem size :" << image.elemSize()<< std::endl;
    std::cout << "image channels : "  << image.channels()<< std::endl;
    std::cout << "image size: " << image.size()<< std::endl;

    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }
    
    // OPTIONAL : 
    // convert Mat to vector
    cv::Mat flat = image.reshape(1, image.total()*image.channels());
    
    // for input vector in host memory 
    std::vector<uchar> vec = image.isContinuous()? flat : flat.clone(); 

    std::cout << "Vector len: " << vec.size() << std::endl;


    // OpenCL setup
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Load and compile the OpenCL kernel
    FILE *kernelFile = fopen(KERNEL_FILE, "r");
    if (!kernelFile) {
        std::cerr << "Error: Could not open kernel file." << std::endl;
        return -1;
    }

    fseek(kernelFile, 0, SEEK_END);
    size_t kernelSize = ftell(kernelFile);
    rewind(kernelFile);

    char *kernelSource = new char[kernelSize + 1];
    fread(kernelSource, 1, kernelSize, kernelFile);
    kernelSource[kernelSize] = '\0';
    fclose(kernelFile);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "rotate_image", nullptr);

    // Set up OpenCL buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(unsigned char) * image.rows * image.cols, image.data, nullptr);

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(unsigned char) * image.rows * image.cols, nullptr, nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &image.cols);
    clSetKernelArg(kernel, 3, sizeof(int), &image.rows);
    
    util::Timer timer;
    double start_time;      // Starting time
    double run_time;        // Timing data

    start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    // Execute the kernel
    size_t globalWorkSize[2] = {static_cast<size_t>(image.cols), static_cast<size_t>(image.rows)};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    
    run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
    results(run_time);

    // Read the result back
    unsigned char *outputData = new unsigned char[image.rows * image.cols];
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(unsigned char) * image.rows * image.cols, outputData, 0, nullptr, nullptr);

    // Create an OpenCV Mat from the result
    cv::Mat rotatedImage(image.rows, image.cols, CV_8UC1, outputData);

    // Display the original and rotated images
    cv::imshow("Original Image", image);
    cv::imshow("Rotated Image", rotatedImage);
    cv::waitKey(0);

    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    delete[] kernelSource;
    delete[] outputData;

    return 0;
}

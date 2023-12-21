#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream> 
#include "util.hpp"
#include "device_picker.hpp"


// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

void results( double run_time)
{

    float mflops = 0.0f;
    
    //mflops = 2.0 * N * N * N/(1000000.0f * run_time);
    printf(" %.5f seconds at %.1f MFLOPS \n",  run_time,mflops);

}

int main(int argc, char *argv[])
{
    std::cout << "---- C++ with OpenCL and OpenCV :" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    #ifdef WITH_OPENCL
        std::cout << "OpenCL is enabled in this build." << std::endl;
    #else
        std::cout << "OpenCL is not enabled in this build." << std::endl;
    #endif

    // Load the image
    cv::Mat image = cv::imread("../RotateImage/dog.jpg");
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
    
    int size;   // Number of elements in each matrix

    double start_time;      // Starting time
    double run_time;        // Timing data
    util::Timer timer;      // Timer

    const int R = image.rows;
    const int C = image.cols;
    size = image.rows * image.cols;
    /*
    // convert Mat to vector
    cv::Mat flat = image.reshape(1, image.total()*image.channels());
    
    // for input vector in host memory 
    std::vector<unsigned char> vec = image.isContinuous()? flat : flat.clone(); 

    std::cout << "Vector len: " << vec.size() << std::endl;
    */

    cl::Buffer d_input, d_output;   // Matrices in device memory 

    // Read the result back
   // unsigned char outputData = new unsigned char[R * C];
    std::vector<unsigned char> outputData(image.total() * image.channels()); 

    
//--------------------------------------------------------------------------------
// Create a context and queue
//--------------------------------------------------------------------------------

    try
    {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("horizontal_mirror.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        //directly copying the Mat to Input Buffer 
         d_input = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(uchar) * image.total() * image.elemSize(), image.ptr());
        
        // use below line to Feed the converted vector as to Input Buffer
       //   d_input = cl::Buffer(context, vec.begin(), vec.end(), true);
    
        // Create a cl::Buffer for output 
          d_output = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * image.total() * image.elemSize()); 

      // Create the kernel functor
 
        cl::make_kernel<cl::Buffer, cl::Buffer,int, int> horizontal_mirror(program, "horizontal_mirror");

        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    // Execute the kernel

       // img dimension is usually denotes as width * height in image file properties

        cl::NDRange global( image.cols, image.rows);

        horizontal_mirror(cl::EnqueueArgs(queue, global),
        d_input, d_output, C, R);

    run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
    results(run_time);

    // copying final result to host
    cl::copy(queue, d_output, outputData.begin(), outputData.end());

     // Create an OpenCV Mat from the result
    cv::Mat rotatedImage(image.rows, image.cols, image.type(), outputData.data()); 

    std::cout << "rotatedImage matrix size: \n rows: " << rotatedImage.rows <<", cols: "<< rotatedImage.cols <<"; type: "<< rotatedImage.type() <<", "<<  std::endl;
    std::cout << "rotatedImage matrix dimension : " << rotatedImage.dims<< std::endl; std::cout << "rotatedImage total : " << rotatedImage.total() << std::endl; 
    std::cout << "rotatedImage elem size :" << rotatedImage.elemSize()<< std::endl;
    std::cout << "rotatedImage channels : "  << rotatedImage.channels()<< std::endl;
    std::cout << "rotatedImage size: " << rotatedImage.size()<< std::endl;

    
    // Concatenate images 
    cv::Mat result;
    cv::vconcat(image, rotatedImage, result);
    cv::Mat result1;
    cv::vconcat(rotatedImage, result, result1);

    // Display the original and rotated images
    cv::imshow("Original Image", image);
    cv::imshow("Horizontal mirror", result1);
    cv::imwrite("original.jpg",image);
    cv::imwrite("horizontal.jpg",rotatedImage);
    cv::imwrite("horizontalAll.jpg",result1);
    cv::waitKey(0);

    }catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    return EXIT_SUCCESS;
}
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
    cv::Mat image = cv::imread("./dog.jpg");
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

    cl::Buffer d_input, d_outputR , d_outputL, d_outputM;   // Matrices in device memory 

    // Read the result back
   // unsigned char outputData = new unsigned char[R * C];
    std::vector<unsigned char> outputDataR(image.total() * image.channels()); 
    std::vector<unsigned char> outputDataL(image.total() * image.channels()); 
    std::vector<unsigned char> outputDataM ((image.cols *3) * image.rows * image.channels(),1); 

    
//--------------------------------------------------------------------------------
// Create a context and queue
//--------------------------------------------------------------------------------

    try
    {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("vertical_mirror.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        //directly copying the Mat to Input Buffer 
         d_input = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(uchar) * image.total() * image.elemSize(), image.ptr());
        
        // use below line to Feed the converted vector as to Input Buffer
       //   d_input = cl::Buffer(context, vec.begin(), vec.end(), true);
    
        // Create a cl::Buffer for output 
        d_outputR = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * image.total() * image.channels()); 
        d_outputL = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * image.total() * image.channels()); 
        d_outputM = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * (image.cols *3) * image.rows  * image.channels()); 

      // Create the kernel functor
 
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> vertical_mirror(program, "vertical_mirror");

        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    // Execute the kernel

       // img dimension is usually denotes as width * height in image file properties

        cl::NDRange global(image.cols, image.rows);
 
        vertical_mirror(cl::EnqueueArgs(queue, global),
        d_input, d_outputR, d_outputL, C, R);

        run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
        results(run_time);

    // copying final result to host
    cl::copy(queue, d_outputR, outputDataR.begin(), outputDataR.end());
    cl::copy(queue, d_outputL, outputDataL.begin(), outputDataL.end());
    cl::copy(queue, d_outputM, outputDataM.begin(), outputDataM.end());

     // Create an OpenCV Mat from the result
    cv::Mat rotatedImageR(image.rows, image.cols, image.type(), outputDataR.data()); 
    cv::Mat rotatedImageL(image.rows, image.cols, image.type(), outputDataL.data()); 
    cv::Mat rotatedImageM(image.rows, (1*image.cols), image.type(), outputDataM.data()); 

    std::cout << "rotatedImage matrix size: \n rows: " << rotatedImageR.rows <<", cols: "<< rotatedImageR.cols <<"; type: "<< rotatedImageR.type() <<", "<<  std::endl;
    std::cout << "rotatedImageR matrix dimension : " << rotatedImageR.dims<< std::endl; std::cout << "rotatedImageR total : " << rotatedImageR.total() << std::endl; 
    std::cout << "rotatedImageR elem size :" << rotatedImageR.elemSize()<< std::endl;
    std::cout << "rotatedImageR channels : "  << rotatedImageR.channels()<< std::endl;
    std::cout << "rotatedImageR size: " << rotatedImageR.size()<< std::endl;

    std::cout << "rotatedImage M matrix size: \n rows: " << rotatedImageM.rows <<", cols: "<< rotatedImageM.cols <<"; type: "<< rotatedImageM.type() <<", "<<  std::endl;
    std::cout << "rotatedImageM matrix dimension : " << rotatedImageM.dims<< std::endl; std::cout << "rotatedImageM total : " << rotatedImageM.total() << std::endl; 
    std::cout << "rotatedImageM elem size :" << rotatedImageM.elemSize()<< std::endl;
    std::cout << "rotatedImageM channels : "  << rotatedImageM.channels()<< std::endl;
    std::cout << "rotatedImageM size: " << rotatedImageM.size()<< std::endl;

    
    // Concatenate images horizontally
    cv::Mat result;
    cv::hconcat(image, rotatedImageR, result);
    cv::Mat result1;
    cv::hconcat(rotatedImageL, result, result1);
    
    // Display the original and rotated images
    cv::imshow("Original Image", image);
     // Display the result
    cv::imshow("Vertical mirrors", result1);
    cv::imshow("Vertical mid", rotatedImageM);

    //cv::imshow("Rotated Image", rotatedImageR);
    cv::imwrite("original.png",image);
    cv::imwrite("verticalR.png",rotatedImageR);
    cv::imwrite("verticalL.png",rotatedImageL);
    cv::imwrite("verticalAll.png",result1);
    
   
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
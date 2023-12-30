#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <array>
#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
//#define LENGTH (2)    // length of vectors a, b, and c

int main(void)
{   
    
    const int M = 3; //depth
    const int N = 4; // height
    const int O = 4; // width

    // pooling window size
    const int window = 2;

    const int stride = 2;  //for pooling, better to match with window size 
    
    const int padding = 0;  // set to zero
    // the current kernel cannot handle padding yet. as padding is used only in odd dimensions or where needed. // mostly right and bottom padding
    // Example input matrix
   
     int input[3][4][4] = {
         {{58, 94, 0, 45}, {45, 42, 33, 56}, {88, 92, 45, 22}, {23, 35, 36, 30}},
        {{36, 85, 94, 36}, {25, 29, 90, 74}, {66, 27, 64, 32}, {78, 91, 98, 75}},
        {{34, 52, 75, 18}, {36, 53, 39, 52}, {27, 77, 82, 96}, {41, 13, 81, 63}}
    };

    const int opM = ((M + 2 * padding - window) / stride) + 1;
    const int opN = ((N + 2 * padding - window) / stride) + 1;
    const int opO = ((O + 2 * padding  - window)/ stride ) + 1;
    std::cout << "output dim: "<< opM <<opN << opO << std::endl;
    std::array<std::array<std::array<int, opO>, opN>, opM> output;

    std::vector<int> h_input(M * N * O);    // input
    
    std::vector<int> h_output(opM * opN * opO);    // convolution output w stride =1, no padding 

    //flatten input to vectors to copy to buffer 

    int i, j, k, index;
     for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
        for (k = 0; k < O; k++) {
            // linear_index=i×(dim2×dim3)+j×dim3+k
            index = (i * (N * O) + j * (O)+ k);
            h_input[index] = input[i][j][k];
          //  std::cout << i << j << k << " " << (index) <<  " "<< input[i][j][k] << std::endl;
        }
        }
    } 
    i=0;j=0;k=0;index=0;
    
    std::cout <<"elem# in inp " << h_input.size() << std::endl;

    cl::Buffer d_a;                        // device memory used for the input  vector
    cl::Buffer d_c;                       // device memory used for the output  vector

     
    try 
    {
        
    	// Create a context
        cl::Context context(DEVICE);
        std::cout<< "Context for Device : " << DEVICE << std::endl;
    
     // Get the command queue
        cl::CommandQueue queue(context);

        // Load in kernel source, creating a program object for the context
        //  cl::Program program(context, util::loadProgram("maxpool3.cl"), true);

    //   /* for checking kernel errors 

        const char* kernelSource = R"( 
            #define W_SIZE  8
            __kernel void maxpool3(                             
            __global int* input,                                         
            __global int* output,                      
            const int M,
            const int N,
            const int O,
            const int stride,
            const int window,
            const int padding)               
        {                                          
            int gidI = get_global_id(0);               
            int gidJ = get_global_id(1);               
            int gidK = get_global_id(2);   

            const int w_size = window * window * window;            
            int inp[W_SIZE];  

            const int opM = ((M + 2 * padding - window) / stride) + 1;
            const int opN = ((N + 2 * padding - window) / stride) + 1;
            const int opO = ((O + 2 * padding  - window)/ stride ) + 1;

            //printf("get id i %d ,j %d , k %d \n", gidI, gidJ, gidK);

            // Adjust the bounds considering padding
            int d = (gidI * stride - padding) + window;
            int h = (gidJ * stride - padding) + window; 
            int w = (gidK * stride - padding) + window; 
            int count = 0;
            int max = INT_MIN; 
              //printf("for gid %d %d %d the limits are %d %d %d \n", gidI, gidJ, gidK, d, h, w);
                for (int m = gidI * stride - padding; m < d; ++m) {
                    for (int n = gidJ * stride - padding; n < h; ++n) {
                        for (int k = gidK * stride - padding; k < w; ++k) {
                            
                            // Check if the indices are within the valid range
                            if (m >= 0 && m < M && n >= 0 && n < N && k >= 0 && k < O) {
                                inp[count] = input[(m * (N * O) + n * O + k)];                              
                                // printf("inp at  %d %d is %d \n", m, n, input[(m * (N * O) + n * O + k)] );
                            } else {
                                // Handle padding by assigning zero or a default value
                                inp[count] = 0;
                            }
                            count = count + 1;
                        }
                    }
                }

                for (int b = 0; b < (window * window * window); ++b) {
                    if(inp[b] > max)
                      max = inp[b];
                    // printf("for gid %d %d %d the elem are %d \n", inp[b]);
                }
                output[gidI * (opN * opO) + gidJ * (opO) + gidK] = max;
              
                // Debugging statement
                // printf("for i %d, j %d, k %d, max %d\n", gidI, gidJ, gidK, max);
        }            
        )";      
         std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

         cl::Program::Sources sources(1, std::make_pair(kernelSource, strlen(kernelSource)));
         cl::Program program(context, sources);
         
          try {
            program.build(devices);
        } catch (const cl::Error& e) {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::cerr << "Build error:" << std::endl;
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        } else {
            std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }
        return 1;
    }
//  */
        // Create the kernel functor
 
        auto maxpool3 = cl::make_kernel<cl::Buffer, cl::Buffer, int, int, int, int,int,int>(program, "maxpool3");

       // d_a   = cl::Buffer(context, begin(h_a), end(h_a), true); 

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * M * N * O, h_input.data());

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (opM * opN * opO) );
               
        util::Timer timer;
        cl::NDRange global(opM, opN, opO); // depth x height x width 
        maxpool3(
            cl::EnqueueArgs(
                queue,
                global), 
            d_a,
            d_c,
            M,
            N,
            O,
            stride,
            window,
            padding);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_output), end(h_output));
        
        std::cout << "input " << M << N  << O << "(ip - fil + 1) output " << opM << opN << opO << std::endl;
        std::cout << "output len " << h_output.size() << std::endl;
        // Test the results 
        for(int g = 0; g < (opM* opN * opO); g++) {
            std::cout << "output " << h_output[g] << " "  << g << std::endl;
        }
      //  */
    }
     catch (cl::Error err) {
       
        std::cout << "Exception\n";
        std::cerr 
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
           << ")"
           << std::endl;
    }
    
}

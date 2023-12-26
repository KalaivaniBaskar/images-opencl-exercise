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
    
    const int M = 6;
    const int N = 6;
    const int O = 3;
    const int stride = 1;
    const int padding = 1;
    // Example input matrix
   
     int input[6][6][3] = {
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
        {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}, {28, 29, 30}, {31, 32, 33}, {34, 35, 36}},
        {{37, 38, 39}, {40, 41, 42}, {43, 44, 45}, {46, 47, 48}, {49, 50, 51}, {52, 53, 54}},
        {{55, 56, 57}, {58, 59, 60}, {61, 62, 63}, {64, 65, 66}, {67, 68, 69}, {70, 71, 72}},
        {{73, 74, 75}, {76, 77, 78}, {79, 80, 81}, {82, 83, 84}, {85, 86, 87}, {88, 89, 90}},
        {{91, 92, 93}, {94, 95, 96}, {97, 98, 99}, {100, 101, 102}, {103, 104, 105}, {106, 107, 108}}
    };

    // Example filter matrix
   
     int filter[3][3][3] = {
        {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}},
        {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}},
        {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}}
    };
    const int opM = ((M + 2 * padding - 3) / stride) + 1;
    const int opN = ((N + 2 * padding - 3) / stride) + 1;
    const int opO = ((O - 3)/ stride ) + 1;
    std::cout << "output dim: "<< opM <<opN << opO << std::endl;
    std::array<std::array<std::array<int, opO>, opN>, opM> output;

    std::vector<int> h_input(M * N * O);    // input
    std::vector<int> h_filter(3 * 3 * 3);    // filter
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
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                // linear_index=i×(dim2×dim3)+j×dim3+k
            index = (i * (3*3) + j * (3)+ k);
            h_filter[index] = filter[i][j][k];
          //  std::cout << i << j << k << " " << (index) << " " << filter[i][j][k] << std::endl;
        }
        }
    } 
    std::cout <<"elem# in inp " << h_input.size() << std::endl;
    std::cout <<"elem# in filter " << h_filter.size() << std::endl;

    cl::Buffer d_a;                        // device memory used for the input  vector
    cl::Buffer d_b;                        // device memory used for the filter  vector
    cl::Buffer d_c;                       // device memory used for the output  vector

     
    try 
    {
        
    	// Create a context
        cl::Context context(DEVICE);
        std::cout<< "Context for Device : " << DEVICE << std::endl;
    
     // Get the command queue
        cl::CommandQueue queue(context);

        // Load in kernel source, creating a program object for the context
        //  cl::Program program(context, util::loadProgram("cnv3.cl"), true);

    //   /* for checking kernel errors 

        const char* kernelSource = R"(
            __kernel void cnv3(                             
            __global int* input,                      
            __global int* filter,                      
            __global int* output,                      
            const int M,
            const int N,
            const int O,
            const int stride,
            const int padding)               
        {                                          
            int gidI = get_global_id(0);               
            int gidJ = get_global_id(1);               
            int gidK = get_global_id(2);               
            int inp[27];  

            const int opM = (M + 2 * padding - 3) / stride + 1;
            const int opN = (N + 2 * padding - 3) / stride + 1;
            const int opO = (O - 3) / stride + 1;

            //printf("get id i %d ,j %d , k %d \n", gidI, gidJ, gidK);

            // Adjust the bounds considering padding
            int h = (gidI * stride - padding) + 3;
            int w = (gidJ * stride - padding) + 3; 
            int d = gidK * stride + 3;
            int count = 0;
            int sum = 0; 

                for (int m = gidI * stride - padding; m < h; ++m) {
                    for (int n = gidJ * stride - padding; n < w; ++n) {
                        for (int k = gidK * stride; k < d; ++k) {
                            
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

                for (int b = 0; b < 27; ++b) {
                    sum = sum + (inp[b] * filter[b]); 
                }
                output[gidI * (opN * opO) + gidJ * (opO) + gidK] = sum;
                // Debugging statement
                // printf("for i %d, j %d, k %d, sum %d\n", gidI, gidJ, gidK, sum);
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
 
        auto cnv3 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int,int>(program, "cnv3");

       // d_a   = cl::Buffer(context, begin(h_a), end(h_a), true); 

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * M * N * O, h_input.data());

        d_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * (3*3*3), h_filter.data());

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (opM * opN * opO) );
               
        util::Timer timer;
        cl::NDRange global(opM, opN, opO);
        cnv3(
            cl::EnqueueArgs(
                queue,
                global), 
            d_a,
            d_b,
            d_c,
            M,
            N,
            O,
            stride,
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

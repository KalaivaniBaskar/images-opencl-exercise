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
#define LENGTH (2)    // length of vectors a, b, and c

int main(void)
{   
    
    const std::size_t M = 5;
    const std::size_t N = 5;
    int size = M * N; 
    // Example input matrix
    std::array<std::array<int, N>, M> input = {{
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    }};

    // Example filter matrix
    std::array<std::array<int, 3>, 3> filter = {{
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    }};

    std::array<std::array<int, N - 2>, M - 2> output;

    std::vector<int> h_input(M * N);    // input
    std::vector<int> h_filter(3 * 3);    // filter
    std::vector<int> h_output((M-2)*(N-2));    // convolution output w stride =1, no padding 

    //flatten input to vectors to copy to buffer 

    int i, j, k;
     for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            h_input[i * N + j] = input[i][j];
          //  std::cout << i << j << " " << (i*N+j) <<  " "<< input[i][j] << std::endl;
        }
    } 
    i =0;j=0;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            h_filter[i * 3 + j] = filter[i][j];
         //  std::cout << i << j << " " << (i*3+j) << " " << filter[i][j]<< std::endl;
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
        //  cl::Program program(context, util::loadProgram("cnv2.cl"), true);

    //   /* for checking kernel errors 

        const char* kernelSource = R"(
            __kernel void cnv2(                             
            __global int* input,                      
            __global int* filter,                      
            __global int* output,                      
            const  int M,
            const  int N)               
            {                                          
            int gidI = get_global_id(0);               
            int gidJ = get_global_id(1);               
            int inp[9];    
            printf("get id i %d ,j %d \n", gidI, gidJ);
            if( gidI < M-2 && gidJ < N-2) {
               int cols = gidI + 2;
               int rows = gidJ + 2; 
               int count = 0;

               for( int m = gidJ; m <= rows; m++ ){
                for(int n = gidI; n <=cols; n++){
                    inp[count] = input[m * M + n]; 
                    count = count + 1;
                    printf("input at %d \n", (m * M + n));
                }
               }
             
               int sum =0;
               for(int k=0; k < 9; k++ ){
                 // printf("inp %d and filt %d \n", inp[k], filter[k]);
                  sum += inp[k] * filter[k];
               } 
               printf("for i %d, j %d , sum %d\n", gidI , gidJ, sum);
               output[gidI *(N-2) + gidJ] = sum;
               printf("output at %d is %d \n", (gidI *(N-2) + gidJ), output[gidI *(N-2) + gidJ] );
            }
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
 
        auto cnv2 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(program, "cnv2");

       // d_a   = cl::Buffer(context, begin(h_a), end(h_a), true); 

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * M * N, h_input.data());

        d_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * 9, h_filter.data());

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * ((M-2) * (N-2)) );

               
        util::Timer timer;
        cl::NDRange global(M-2, N-2);
        cnv2(
            cl::EnqueueArgs(
                queue,
                global), 
            d_a,
            d_b,
            d_c,
            M,
            N);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_output), end(h_output));
        
        // Test the results 
        for(int g = 0; g < ((M-2 )* (N-2)); g++) {
            std::cout << "output " << h_output[g] << " " << std::endl;
        }
        
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

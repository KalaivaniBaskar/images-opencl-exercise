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
    
    const int M = 7;
    const int N = 7;
    int size = M * N; 
    int stride = 2;
    // Example input matrix
    std::array<std::array<int, N>, M> input = {{
        {1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19, 20, 21},
        {22, 23, 24, 25, 26, 27, 28},
        {29, 30, 31, 32, 33, 34, 35},
        {36, 37, 38, 39, 40, 41, 42},
        {43, 44, 45, 46, 47, 48, 49},
    }};

    // Example filter matrix
    std::array<std::array<int, 3>, 3> filter = {{
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    }};
    int opM = ((M - 3)/ stride ) + 1;
    int opN = ((N - 3)/ stride ) + 1;
    std::array<std::array<int, 3>, 3> output;

    std::vector<int> h_input(M * N);    // input
    std::vector<int> h_filter(3 * 3);    // filter
    std::vector<int> h_output((opM)*(opN));    // convolution output w stride =1, no padding 

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
    std::cout <<"elem# op " << opM << " x " << opN << std::endl;
    std::cout << "stride is " << stride << std::endl;
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
            const  int N,
            const int stride)               
            {                                          
            int gidI = get_global_id(0);               
            int gidJ = get_global_id(1);        
            int opM = ((M - 3)/ stride ) + 1;
            int opN = ((N - 3)/ stride ) + 1;       
            int inp[9];    
            printf("get id i %d ,j %d \n", gidI, gidJ);
            if( gidI < opM && gidJ < opN) {
               int cols = gidI * stride + 3;
               int rows = gidJ * stride + 3; 
               int count = 0;

               for( int m = gidJ * stride; m < rows; m++ ){
                for(int n = gidI * stride; n < cols; n++){
                    inp[count] = input[m * M + n]; 
                    count = count + 1;
                    printf("input at %d is %d\n", (m * M + n), input[m * M + n]);
                }
               }
             
               int sum =0;
               for(int k=0; k < 9; k++ ){
                 // printf("inp %d and filt %d \n", inp[k], filter[k]);
                  sum += inp[k] * filter[k];
               } 
               printf("for i %d, j %d , sum %d\n", gidI , gidJ, sum);
               output[gidI * opN + gidJ] = sum;
               printf("output at %d is %d \n", (gidI * opN + gidJ), output[gidI * opN + gidJ] );
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
 
        auto cnv2 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int>(program, "cnv2");

       // d_a   = cl::Buffer(context, begin(h_a), end(h_a), true); 

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * M * N, h_input.data());

        d_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * 9, h_filter.data());

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * ((opM) * (opN)) );

               
        util::Timer timer;
        cl::NDRange global(opM, opN);
        cnv2(
            cl::EnqueueArgs(
                queue,
                global), 
            d_a,
            d_b,
            d_c,
            M,
            N,
            stride);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_output), end(h_output));
        
        std::cout << "input " << M << N << "(ip - fil + 1) output " << M-2 <<N-2 << std::endl;
        std::cout << "output len " << h_output.size() << std::endl;
        // Test the results 
        for(int g = 0; g < ((opM )* (opN)); g++) {
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

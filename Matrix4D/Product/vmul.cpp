#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

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
    int size = LENGTH; 
     // a vector of 4 dimensions vec[size][size][size][size]
    // Initialization of a 4D vector with values
    std::vector<std::vector<std::vector<std::vector<int> > > > input_a = {
        {
            {
                {5, 3}, {5, 3}
            }, 
            {
                {6, 7}, {6, 7}
            }
        },
        {
            {
                {8, 9}, {8, 9}
            },
            {
                {9, 7}, {9, 7}
            }
        }
    };
    // b vector 
    // Initialization of a 4D vector with values
    std::vector<std::vector<std::vector<std::vector<int> > > > input_b = {
        {
            {
                {15, 13}, {15, 13}
            }, 
            {
                {16, 17}, {16, 71}
            }
        },
        {
            {
                {18, 19}, {18, 19}
            },
            {
                {91, 71}, {91, 17}
            }
        }
    };
    // std::vector<std::vector<std::vector<std::vector<int> > > > h_c;    // c = a + b, from compute device
    std::vector<int> h_a(size * size * size * size);    // a
    std::vector<int> h_b(size * size * size * size);    // b
    std::vector<int> h_c(size * size * size * size);    // c = a + b, from compute device

    //flatten input 4D vectors to copy to buffer 

    int i, j, k, l;
     for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            for (k = 0; k < size; k++) { 
                for (l = 0; l < size; l++) {  
                  h_a[i * (size * size * size) + j * (size * size) + k * size + l] = input_a[i][j][k][l];
                  h_b[i * (size * size * size) + j * (size * size) + k * size + l] = input_b[i][j][k][l];
                }
            }
        }
    } 

    cl::Buffer d_a;                        // device memory used for the input  a vector
    cl::Buffer d_b;                        // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the output c vector

    // Fill vectors a and b with random float values
    // int count = LENGTH;
    // for(int i = 0; i < count; i++)
    // {
    //     h_a[i]  = rand() / (float)RAND_MAX;
    //     h_b[i]  = rand() / (float)RAND_MAX;
    // }
      std::cout << "A: "<< input_a[0][0][0][0] << " h " << h_a[0] << h_b[1] << "\tB: " << input_b[1][1][1][1] << std::endl;
    try 
    {
    	// Create a context
        cl::Context context(DEVICE);
        std::cout<< "Context for Device : " << DEVICE << std::endl;
    
     // Get the command queue
        cl::CommandQueue queue(context);

        // Load in kernel source, creating a program object for the context
        //  cl::Program program(context, util::loadProgram("vmul.cl"), true);

    //   /* for checking kernel errors 

        const char* kernelSource = R"(
            __kernel void vmul(                             
            __global int* a,                      
            __global int* b,                      
            __global int* c,                      
            const unsigned int size)               
            {                                          
            int gidI = get_global_id(0);               
            int gidJ = get_global_id(1);               
            int gidK = get_global_id(2);      
            int l;   
            int index;       
        
            if( gidI < size && gidJ < size && gidK < size ) {
                for(l=0; l < size; l++){
                    index = gidI * (size * size * size) + gidJ * (size * size) + (gidK * size) + l;
                    printf("ids %d %d %d \n", gidI, gidJ, gidK);
                    printf("index %d  \n", index);
                    c[index] = a[index] * b[index];
                }   
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
//*/
        // Create the kernel functor
 
        auto vmul = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vmul");

       // d_a   = cl::Buffer(context, begin(h_a), end(h_a), true); 

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * size*size*size*size, h_a.data());

        d_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * size*size*size*size, h_b.data());

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (size * size * size * size));

               
        util::Timer timer;
        cl::NDRange global(size,size,size);
        vmul(
            cl::EnqueueArgs(
                queue,
                global), 
            d_a,
            d_b,
            d_c,
            size);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_c), end(h_c));
          
        //   // Enqueue a command to read data from the device to the host
        //   std::vector<int> hostArray(16);
        //   cl::copy(queue, d_c, hostArray.begin(), hostArray.end());
        //     // Print the data
        //     for (int i = 0; i < 16; ++i) {
        //         std::cout << h_a[i] <<" + " << h_b[i] << " = " << hostArray[i] << " ; \n";
        //     }

        // Test the results 
        for(int g = 0; g < (size * size * size * size); g++) {
            std::cout << h_a[g] <<" * " << h_b[g] << " = " << h_c[g] << " " << std::endl;
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

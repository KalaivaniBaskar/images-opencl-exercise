#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (128)    // length of vectors a, b, and c

int main(void)
{   
    int size = LENGTH; 
    int localsize = 64; 
    int group = size/localsize;

    /*     // a vector of 4 dimensions vec[size][size][size][size]
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
    std::vector<int> h_a;    // a
    h_a.resize(size);
    // Assigns value 0 to all the elements in the vector
    std::fill(h_a.begin(), h_a.end(), 22); 
    // 0 0 0 0 0 0
    h_a[995] = 75;
    */

   std::vector<int> h_a = {
        43, 12, 65, 78, 29, 41, 54, 92, 17, 88,
        5, 36, 71, 49, 23, 68, 10, 37, 81, 95,
        14, 77, 30, 52, 19, 63, 82, 45, 26, 58,
        74, 8, 97, 61, 33, 70, 22, 50, 93, 16,
        67, 38, 84, 11, 76, 27, 59, 89, 44, 69,
        31, 53, 20, 72, 7, 85, 98, 15, 60, 34,
        87, 24, 51, 73, 6, 42, 99, 18, 46, 79,
        28, 55, 96, 13, 66, 39, 21, 80, 64, 48,
        91, 25, 62, 35, 90, 47, 75, 9, 56, 32,
        83, 40, 94, 57, 90, 47, 75, -9, 23, 32,
        28, 55, 96, 13, 66, 39, 21, 80, 64, 48,
        28, 55, 96, 13, 66, 39, 21, 80, 64, 48,
        28, 55, 96, 13, 66, 39, 21, 80
    };
    std::vector<int> h_c(group);    
     
    cl::Buffer d_a;      // device memory used for the input  a vector
    cl::Buffer d_c;        // device memory used for the output c vector

    // Fill vectors a and b with random float values
    // int count = LENGTH;
    // for(int i = 0; i < count; i++)
    // {
    //     h_a[i]  = rand() / (float)RAND_MAX;
    //     h_b[i]  = rand() / (float)RAND_MAX;
    // }
      std::cout <<  " h " << h_a[0]  << "  "<< h_a[155]<< std::endl;
    try 
    {
    	// Create a context
        cl::Context context(DEVICE);
        std::cout<< "Context for Device : " << DEVICE << std::endl;
    
     // Get the command queue
        cl::CommandQueue queue(context);

        // Load in kernel source, creating a program object for the context
        //  cl::Program program(context, util::loadProgram("vmax.cl"), true);

    //   /* for checking kernel errors 

        const char* kernelSource = R"(
            #define WORK_GROUP_SIZE 64
            
    __kernel void vmax(__global const int* inputArray, __global int* result, const unsigned int size) {
        // Allocate local memory for each work-group
        __local int localMaxBuffer[WORK_GROUP_SIZE];

        // Get global and local IDs
        const int globalID = get_global_id(0);
        const int localID = get_local_id(0);
        const int groupID = get_group_id(0);

        // Initialize localMaxBuffer with the element of the work-group
        localMaxBuffer[localID] = inputArray[globalID];

        // Synchronize to make sure all elements are loaded into local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform parallel reduction within the work-group to find the maximum value
        for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
            if (localID < stride) {
                // Compare and update localMaxBuffer
                localMaxBuffer[localID] = max(localMaxBuffer[localID], localMaxBuffer[localID + stride]);
            }
            // Synchronize to make sure all threads have updated their values
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // The first thread of each work-group writes the local maximum to global memory
        if (localID == 0) {
            result[groupID] = localMaxBuffer[0];
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
 
        auto vmax = cl::make_kernel<cl::Buffer, cl::Buffer, int>(program, "vmax");

       // d_a   = cl::Buffer(context, begin(h_a), end(h_a), true); 

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * h_a.size(), h_a.data());

      //  d_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * size*size*size*size, h_b.data());

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (group));

               
        util::Timer timer;
        cl::NDRange global(size);
        cl::NDRange local(localsize);
        vmax(
            cl::EnqueueArgs(
                queue,
                global,
                local), 
            d_a,
            d_c,
            h_a.size());

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_c), end(h_c));
         
           // Input
        std::cout <<  "\n the input \n" << h_a.size()<< std::endl;
        for(int f = 0; f < h_a.size(); f++) {
            std::cout << h_a[f] << " ";
        } 
        std::cout <<  "\n the output \n" << std::endl;

        // Test the results 
        for(int g = 0; g < (group); g++) {
            std::cout << "at index " << g << "is " << h_c[g] << " " << std::endl;
        } 

        // Select the element with the minimum value
            auto it = std::max_element(h_a.begin(), h_a.end());
            // Check if iterator is not pointing to the end of vector
            if(it != h_a.end())
            {
                std::cout<<"Testing Maximum Value: "<<*it << std::endl;
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

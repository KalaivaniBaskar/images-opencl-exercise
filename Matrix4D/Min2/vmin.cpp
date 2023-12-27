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
     // a vector of 4 dimensions vec[size][size][size][size]
    // Initialization of a 4D vector with values
   
    // b vector 
    // Initialization of a 4D vector with values
   
    // std::vector<std::vector<std::vector<std::vector<int> > > > h_c;    // c = a + b, from compute device
   
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
        83, 40, 94, 57, 90, 47, 75, -9, 56, 32,
        28, 55, 96, 13, 66, 39, 21, 80, 64, 48,
        28, 55, 96, 13, 66, 39, 21, 80, 64, 48,
        28, 55, 96, 13, 66, 39, 21, 80
    }; // a
    /*
    h_a.resize(size);
    // Assigns value 0 to all the elements in the vector
    std::fill(h_a.begin(), h_a.end(), 88); 
    // 0 0 0 0 0 0
    //h_a[155] = 12;
    h_a[9] = 7;
    */

    std::vector<int> h_c(group);    
     
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
      std::cout <<  " h " << h_a.size()  << " " << h_a[64] << std::endl;
    try 
    {
    	// Create a context
        cl::Context context(DEVICE);
        std::cout<< "Context for Device : " << DEVICE << std::endl;
    
     // Get the command queue
        cl::CommandQueue queue(context);

        // Load in kernel source, creating a program object for the context
        //  cl::Program program(context, util::loadProgram("find_min.cl"), true);

    //   /* for checking kernel errors 

            const char* kernelSource = R"(
            #define WORK_GROUP_SIZE 64

            __kernel void find_min(__global const int* input, __global int* output, const int size) {
                __local int localMin[WORK_GROUP_SIZE];

                int globalID = get_global_id(0);
                int localID = get_local_id(0);
                int groupID = get_group_id(0);
                int groupSize = get_local_size(0);

                int localIndex = localID;
                int globalIndex = globalID;

                // Initialize local min to a large value
                localMin[localID] = INT_MAX; 
                // Use FLT_MAX for C++ or CL_FLT_MAX for OpenCL

                barrier(CLK_LOCAL_MEM_FENCE);

                // Find the minimum within the work group
                while (globalIndex < size) {
                    if(input[globalIndex] < localMin[localID]){
                         localMin[localID] = input[globalIndex];
                         printf("here ");
                    }
                    globalIndex += get_global_size(0);
                }
                printf("localMin[] %d is  %d\n", localID, localMin[localID]);

                barrier(CLK_LOCAL_MEM_FENCE);

                // Perform the reduction within the work group using tree reduction
                for (int stride = groupSize / 2; stride > 0; stride /= 2) {
                    if (localIndex < stride) {            
                       
                        // printf("localmin %d and stride %d at group %d\n", localMin[localIndex], localMin[localIndex + stride], groupID);         
                       
                         if(localMin[localIndex + stride] < localMin[localIndex]){
                         localMin[localIndex] = localMin[localIndex + stride];
                    }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                // Write the result to global memory
                if (localIndex == 0) {
                    output[groupID] = localMin[0];
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
 
        auto find_min = cl::make_kernel<cl::Buffer, cl::Buffer, int>(program, "find_min");

       // d_a   = cl::Buffer(context, begin(h_a), end(h_a), true); 

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * size, h_a.data());

      //  d_b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(int) * size*size*size*size, h_b.data());

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (group));

               
        util::Timer timer;
        cl::NDRange global(size);
        cl::NDRange local(localsize);
        find_min(
            cl::EnqueueArgs(
                queue,
                global,
                local), 
            d_a,
            d_c,
            size);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_c), end(h_c));
         
          // Input
        for(int f = 0; f < (size); f++) {
            std::cout << h_a[f] << " ";
        } 
        std::cout <<  "\n above input \n" << std::endl;
        // Test the results 
        for(int g = 0; g < (group); g++) {
            std::cout << "at index " << g << "is " << h_c[g] << " " << std::endl;
        } 
        // Select the element with the minimum value
            auto it = std::min_element(h_a.begin(), h_a.end());
            // Check if iterator is not pointing to the end of vector
            if(it != h_a.end())
            {
                std::cout<<"Testing Minimum Value: "<<*it << std::endl;
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

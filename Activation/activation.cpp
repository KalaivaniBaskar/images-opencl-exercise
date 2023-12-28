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

#define LENGTH (128)    // length of vectors a, b, and c

/*
Activation function and their formula 

 relu(x):
    output is maximum(0, x)

 sigmoid(x):
    output = 1 / (1 + exp(-x))

 relu6(x):
    output is minimum(maximum(0, x), 6)

 gelu(x):
    output = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))


 silu(x):
    output = x * sigmoid(x)

 tanh(x):
    output is tanh(x)
*/
int main(void)
{   
    int size = LENGTH; 
    int localsize = 64; 
    int group = size/localsize;


   std::vector<float> h_a = {1.0, -2.0, 3.0, -4.0, 5.0, 10.0};
    std::vector<float> h_relu(h_a.size());    
    std::vector<float> h_relu6(h_a.size());  
    std::vector<float> h_gelu(h_a.size());  
    std::vector<float> h_silu(h_a.size());  
    std::vector<float> h_sigmoid(h_a.size());  
    std::vector<float> h_tanh(h_a.size());    
    std::vector<float> h_softmax(h_a.size());    
     
    cl::Buffer d_a;      // device memory used for the input  a vector
    cl::Buffer d_relu, d_relu6, d_gelu, d_silu, d_sigmoid, d_tanh, d_softmax;        // device memory used for the output c vector

    try 
    {
    	// Create a context
        cl::Context context(DEVICE);
        std::cout<< "Context for Device : " << DEVICE << std::endl;
    
     // Get the command queue
        cl::CommandQueue queue(context);

        // Load in kernel source, creating a program object for the context
        //  cl::Program program(context, util::loadProgram("activation.cl"), true);

    //   /* for checking kernel errors 

        const char* kernelSource = R"(
            
    __kernel void activation(__global const float* input, __global float* relu, __global float* relu6, __global float* gelu, __global float* silu, __global float* sigmo, __global float* tanh_op, __global float* softmax, const unsigned int size) {

        // Get global and local IDs
        const int gid = get_global_id(0);

        if( gid < size){

            relu[gid] = fmax( 0.0f, input[gid]); 

            relu6[gid] = fmin( fmax(0.0f, input[gid]), 6.0f );

            gelu[gid] = 0.5f * input[gid] * (1 + tanh(sqrt(2.0f / M_PI) * (input[gid] + 0.044715f * (input[gid]*input[gid]*input[gid]) )));

            silu[gid] = input[gid] * (1 / (1 + exp(-input[gid])));

            sigmo[gid] = 1 / (1 + exp(-input[gid])); 

            tanh_op[gid] = tanh(input[gid]); 

            // Compute sum of exp(x) of the entire vector this can be assigned to localID = 0 workitem and using barrier 
            // so that the loop is runs only once for each work-group in case of large datasets. 
            float exp_sum = 0.0;
            for (int i = 0; i < size; ++i) {
                exp_sum += exp(input[i]);
            }

            // Compute softmax for the current element - this can be done by each work item
            softmax[gid] = exp(input[gid]) / exp_sum;

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
 
        auto activation = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "activation");

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(float) * h_a.size(), h_a.data());

        d_relu  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * h_a.size());
        d_relu6  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * h_a.size());
        d_gelu  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * h_a.size());
        d_silu  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * h_a.size());
        d_sigmoid  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * h_a.size());
        d_tanh  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * h_a.size());
        d_softmax  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * h_a.size());
              
        util::Timer timer;
        cl::NDRange global(h_a.size());

        activation(
            cl::EnqueueArgs(
                queue,
                global), 
            d_a,
            d_relu,
            d_relu6,
            d_gelu,
            d_silu,
            d_sigmoid,
            d_tanh,
            d_softmax,
            h_a.size());

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_relu, begin(h_relu), end(h_relu));
        cl::copy(queue, d_relu6, begin(h_relu6), end(h_relu6));
        cl::copy(queue, d_gelu, begin(h_gelu), end(h_gelu));
        cl::copy(queue, d_silu, begin(h_silu), end(h_silu));
        cl::copy(queue, d_sigmoid, begin(h_sigmoid), end(h_sigmoid));
        cl::copy(queue, d_tanh, begin(h_tanh), end(h_tanh));
        cl::copy(queue, d_softmax, begin(h_softmax), end(h_softmax));
         
           // Input
        std::cout <<  "\n the input \n" << h_a.size()<< std::endl;
        for(int f = 0; f < h_a.size(); f++) {
            std::cout << h_a[f] << " ";
        } 
        std::cout <<  "\n the outputs: " << std::endl;

        // Test the results 
        for(int g = 0; g < (h_a.size()); g++) {
            std::cout << "Inp: " << h_a[g]<< "   Relu: " << h_relu[g] << "   Relu6: "<< h_relu6[g] << "   Gelu: " << h_gelu[g] << "   Silu: " << h_silu[g] << "    Sigmoid: " << h_sigmoid[g] << "   Tanh: " << h_tanh[g]  << "   Softmax: " << h_softmax[g]  << std::endl;
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

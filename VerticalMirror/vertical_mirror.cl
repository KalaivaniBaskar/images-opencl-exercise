__kernel void vertical_mirror(__global const uchar *input, __global uchar *output, int width, int height) {
    int i = get_global_id(0);
    // i denotes no . of rows, so img height
   
    int j = get_global_id(1);
    // j denotes no of. cols, so img width 

    // img dim is usually width * height 
    
    int tmp;
    if (i < height && j < (width / 2)) {
        tmp = input[i * width + j];
        output[i* width + j] = input[i* width + (width -j - 1)];
        output[i* width + (width -j - 1)] = tmp;       
    }
}

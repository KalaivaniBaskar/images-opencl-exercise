__kernel void horizontal_mirror(__global const uchar *input, __global uchar *output, int width, int height) {
    int i = get_global_id(0);
    // i denotes no . of cols, so img width
   
    int j = get_global_id(1);
    // j denotes no of. rows, so img height 

    // img dim is usually width * height 
    
    int tmp;
    if (i < (width) && j < (height/2)) {
       int bottomIndex = (height - j - 1) * width + i;
       int topIndex  = j * width + i;

        for (int channel = 0; channel < 3; ++channel) {
            output[bottomIndex * 3 + channel] = input[topIndex * 3 + channel];
            output[topIndex * 3 + channel] = input[bottomIndex * 3 + channel];
        }    
    }
}

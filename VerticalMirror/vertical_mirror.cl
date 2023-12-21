__kernel void vertical_mirror(__global const uchar *input, __global uchar *outputR, __global uchar *outputL, int width, int height) {
   int i = get_global_id(0);
    // i denotes no . of cols, so img width
   
    int j = get_global_id(1);
    // j denotes no of. rows, so img height 
    // img dim is usually width * height 
    
    if (i < (width / 2) && j < height) {
        int leftIndex = j * width + i;
        int rightIndex = j * width + (width - i - 1);

        for (int channel = 0; channel < 3; ++channel) {
            outputR[leftIndex * 3 + channel] = input[rightIndex * 3 + channel];
            outputR[rightIndex * 3 + channel] = input[leftIndex * 3 + channel]; 

            // Left-side mirror
            outputL[rightIndex * 3 + channel] = input[leftIndex * 3 + channel];
            outputL[leftIndex * 3 + channel] = input[rightIndex * 3 + channel];
        }
    } 

} 
__kernel void rotate_image(__global const uchar *input, __global uchar *output, int width, int height) {
    int gidX = get_global_id(0);
    int gidY = get_global_id(1);

    if (gidX < width && gidY < height) {
        int newGidX = gidX;
        int newGidY = gidY;

        // Rotation logic - example rotates image by given angle
        float angle = 60.0f;
        float radians = angle * 3.14159265358979323846 / 180.0;
        int centerX = width / 2;
        int centerY = height / 2;

        newGidX = (int)((gidX - centerX) * cos(radians) - (gidY - centerY) * sin(radians) + centerX);
        newGidY = (int)((gidX - centerX) * sin(radians) + (gidY - centerY) * cos(radians) + centerY);

        if (newGidX >= 0 && newGidX < width && newGidY >= 0 && newGidY < height) {
            output[newGidY * width + newGidX] = input[gidY * width + gidX];
        }
    }
}

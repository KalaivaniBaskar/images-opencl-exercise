__kernel void rotate_image(__global const uchar *input, __global uchar *output, int width, int height) {
    int gidX = get_global_id(0);
    int gidY = get_global_id(1);

    if (gidX < width && gidY < height) {
        float newGidX = gidX;
        float newGidY = gidY;

        // Rotation logic - example rotates image by given angle
        float angle = 60.0f;
        float radians = angle * 3.14159265358979323846 / 180.0;
        int centerX = width / 2;
        int centerY = height / 2;

        newGidX = ((gidX - centerX) * cos(radians) - (gidY - centerY) * sin(radians) + centerX);
        newGidY = ((gidX - centerX) * sin(radians) + (gidY - centerY) * cos(radians) + centerY);

         // Rounding using rint
        int xRound = rint(newGidX);
        int yRound = rint(newGidY);

        if (xRound >= 0 && xRound < width && yRound >= 0 && yRound < height) {
            output[gidY * width + gidX] = input[yRound * width + xRound];
        }
    }
}

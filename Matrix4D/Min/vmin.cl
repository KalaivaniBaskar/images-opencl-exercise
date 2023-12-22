__kernel void vmin(__global const float* inputArray, __global float* result, const unsigned int size) {
    // Allocate local memory for each work-group
    __local float localMin;

    // Get global and local IDs
    const int globalID = get_global_id(0);
    const int localID = get_local_id(0);
    const int groupID = get_group_id(0);

    // Initialize localMin with the first element of the work-group
    if (localID == 0) {
        localMin = inputArray[globalID];
    }

    // Synchronize to make sure all elements are loaded into local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find the minimum value within the work-group
    for (int i = localID; i < size; i += get_local_size(0)) {
        if (inputArray[i] < localMin) {
            localMin = inputArray[i];
        }
    }

    // Synchronize to make sure all threads in the work-group have finished
    barrier(CLK_LOCAL_MEM_FENCE);

    // The first thread of each work-group writes the local minimum to global memory
    if (localID == 0) {
        result[groupID] = localMin;
    }
}

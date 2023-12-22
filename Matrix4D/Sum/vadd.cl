__kernel void vadd(                             
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
                   // printf("ids %d %d %d \n", gidI, gidJ, gidK);
                   // printf("index %d  \n", index);
                    c[index] = a[index] + b[index];
                }   
            }
            }          
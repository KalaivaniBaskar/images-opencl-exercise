__kernel void cnv2(                             
            __global int* input,                      
            __global int* filter,                      
            __global int* output,                      
            const  int M,
            const  int N)               
            {                                          
            int gidI = get_global_id(0);               
            int gidJ = get_global_id(1);               
            int inp[9];    
            printf("get id i %d ,j %d \n", gidI, gidJ);
            if( gidI < M-2 && gidJ < N-2) {
               int cols = gidI + 2;
               int rows = gidJ + 2; 
               int count = 0;

               for( int m = gidJ; m <= rows; m++ ){
                for(int n = gidI; n <=cols; n++){
                    inp[count] = input[m * M + n]; 
                    count = count + 1;
                    printf("input at %d \n", (m * M + n));
                }
               }
             
               int sum =0;
               for(int k=0; k < 9; k++ ){
                 // printf("inp %d and filt %d \n", inp[k], filter[k]);
                  sum += inp[k] * filter[k];
               } 
               printf("for i %d, j %d , sum %d\n", gidI , gidJ, sum);
               output[gidI *(N-2) + gidJ] = sum;
               printf("output at %d is %d \n", (gidI *(N-2) + gidJ), output[gidI *(N-2) + gidJ] );
            }
            }                               
__kernel void vadd(                             
   __global int* a,                      
   __global int* b,                      
   __global int* c,                      
   const unsigned int size)               
{                                          
   int gid = get_global_id(0);               
   if( gid < (4 * size)) {
            print("index %d", gid );
   }   
}                               
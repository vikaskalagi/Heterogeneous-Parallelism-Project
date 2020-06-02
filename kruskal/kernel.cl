#define MAX_LOCAL_SIZE 256

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable


__kernel void test_kruskal(__global int* src,__global uint * p,__global uint* indicate,const uint size,__global int*dest)
{   //int size=2;
    //int siz=size;
    int gid = get_global_id(0);
    int v;
    int i;

    if(gid<size && indicate[gid]!=2){
            //sorce parent
            v=src[gid]; 
                // to find root of sorce vertex
                while(p[v] != v)
                        {v = p[v];}

                    i = v;
                   v=dest[gid]; 
                   //to find root of destination vertex
                while(p[v] != v)
                        {v = p[v];}
                   int  j = v;
      
                   if(i==j)
                   {
                        indicate[gid]=2;

           
                   }
}
}

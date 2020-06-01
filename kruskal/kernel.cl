#define MAX_LOCAL_SIZE 256

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
typedef struct node
{
    unsigned int src;
    unsigned dest;
   unsigned weight;
   
}node;
int cmp_constant_global(const __global int* a,const __global int* b)
{   //printf("\n\n kernel %d %d \n\n",a[0],b[0]);
    return  (a[0] > b[0]) ? 1 : 0;
}
int cmp_local(__local int *a, __local int *b)
{
    return  (a[0]> b[0] )? 1 : 0;
}

int cmp(int *a, int *b)
{
    return (a[0] > b[0]) ? 1 : 0;
}

inline void swap(int *a, int *b) {
    int tmp;
    tmp = *b;
    *b = *a;
    *a = tmp;
}

// dir == 1 means ascending
inline void sort(int *a, int *b, char dir) {
    if (cmp(a , b) == dir) swap(a, b);
}

inline void swapLocal(__local int *a, __local int *b) {
    int tmp;
    tmp = *b;
    *b = *a;
    *a = tmp;
}

// dir == 1 means ascending
inline void sortLocal(__local int *a, __local int *b, char dir) {
    if (cmp_local(a , b) == dir) swapLocal(a, b);
}


// basic kernel for mergesort start
__kernel void Sort_MergesortStart(const __global int* inArray, __global int* outArray,const __global int* inArray_index, __global int* outArray_index)
{
    __local int local_buffer[2][MAX_LOCAL_SIZE * 2];

    __local int local_buffer_index[2][MAX_LOCAL_SIZE * 2];
    const uint lid = get_local_id(0);
    const uint index = get_group_id(0) * (MAX_LOCAL_SIZE * 2) + lid;
    char pong = 0;
    char ping = 1;

    // load into local buffer
    local_buffer[0][lid] = inArray[index];
    local_buffer[0][lid + MAX_LOCAL_SIZE] = inArray[index + MAX_LOCAL_SIZE];

    local_buffer_index[0][lid] = inArray_index[index];
    local_buffer_index[0][lid + MAX_LOCAL_SIZE] = inArray_index[index + MAX_LOCAL_SIZE];


    // merge sort
    for (unsigned int stride = 2; stride <= MAX_LOCAL_SIZE * 2; stride <<= 1) {
        ping = pong;
        pong = 1 - ping;
        uint leftBoundary = lid * stride;
        uint rightBoundary = leftBoundary + stride;

        uint middle = leftBoundary + (stride >> 1);
        uint left = leftBoundary, right = middle;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (rightBoundary > MAX_LOCAL_SIZE * 2) continue;
#pragma unroll
        for (uint i = 0; i < stride; i++) {
            int leftVal = local_buffer[ping][left];
            int rightVal = local_buffer[ping][right];
            bool selectLeft = left < middle && (right >= rightBoundary || cmp(&leftVal , &rightVal));

            local_buffer[pong][leftBoundary + i] = (selectLeft) ? leftVal : rightVal;

             local_buffer_index[pong][leftBoundary + i] = (selectLeft) ? local_buffer_index[ping][left] : local_buffer_index[ping][right];

            left += selectLeft;
            right += 1 - selectLeft;
        }
    }

    //write back
    barrier(CLK_LOCAL_MEM_FENCE);
    outArray[index] = local_buffer[pong][lid];

    outArray_index[index] = local_buffer_index[pong][lid];

    outArray[index + MAX_LOCAL_SIZE] = local_buffer[pong][lid + MAX_LOCAL_SIZE];

     outArray_index[index + MAX_LOCAL_SIZE] = local_buffer_index[pong][lid + MAX_LOCAL_SIZE];
}

// For smaller strides so we can use local_buffer without getting into memory problems
__kernel void Sort_MergesortGlobalSmall(const __global int* inArray, __global int* outArray, const uint stride, const uint size,const __global int* inArray_index, __global int* outArray_index)
{
    __local int local_buffer[MAX_LOCAL_SIZE * 2];

    __local int local_buffer_index[MAX_LOCAL_SIZE * 2];

    // within one stride merge the different parts
    const uint baseIndex = get_global_id(0) * stride;
    const uint baseLocalIndex = get_local_id(0) * 2;

    uint middle = baseIndex + (stride >> 1);
    uint left = baseIndex;
    uint right = middle;
    bool selectLeft = false;

    if ((baseIndex + stride) > size) return;

    local_buffer[baseLocalIndex + 1] = inArray[left];

    local_buffer_index[baseLocalIndex + 1] = inArray_index[left];

#pragma unroll
    for (uint i = baseIndex; i < (baseIndex + stride); i++) {
        // check which value should be written out
        local_buffer[baseLocalIndex + (int)selectLeft] = (selectLeft) ? inArray[left] : inArray[right];

        local_buffer_index[baseLocalIndex + (int)selectLeft] = (selectLeft) ? inArray_index[left] : inArray_index[right];
        
        selectLeft = left < middle && (right == (baseIndex + stride) || cmp_local(&local_buffer[baseLocalIndex + 1] , &local_buffer[baseLocalIndex]));

        // write out
        outArray[i] = (selectLeft) ? local_buffer[baseLocalIndex + 1] : local_buffer[baseLocalIndex]; //PROBLEMATIC PART! WE RUN OUT OF MEMORY

        outArray_index[i] = (selectLeft) ? local_buffer_index[baseLocalIndex + 1] : local_buffer_index[baseLocalIndex]; 
        //increase counter accordingly
        left += selectLeft;
        right += 1 - selectLeft;
    }
}

__kernel void Sort_MergesortGlobalBig(const __global int* inArray, __global int* outArray, const uint stride, const uint size,const __global int* inArray_index, __global int* outArray_index)
{
    //Problems: Breaks at large arrays. this version was stripped down (so little less performance but supports little bigger arrays)

    // within one stride merge the different parts
    const uint baseIndex = get_global_id(0) * stride;
    const char dir = 1;

    uint middle = baseIndex + (stride >> 1);
    uint left = baseIndex;
    uint right = middle;
    bool selectLeft;

    if ((baseIndex + stride) > size) return;

#pragma unroll
    for (uint i = baseIndex; i < (baseIndex + stride); i++) {
        // check which value should be written out
        selectLeft = (left < middle && (right == (baseIndex + stride) || cmp_constant_global(&inArray[left] , &inArray[right]))) == dir;

        // write out
        outArray[i] = (selectLeft) ? inArray[left] : inArray[right];
         outArray_index[i] = (selectLeft) ? inArray_index[left] : inArray_index[right];

        //increase counter accordingly
        left += selectLeft;
        right += 1 - selectLeft;
    }
}
/*
__kernel void test_kruskal(__global int* inArray,__global uint * p,__global uint* indicate,__global int* outArray,const uint size)
{   //int size=2;
    int siz=size;
    int gid = get_global_id(0);
    int v;
    int i;
    if(gid<siz && indicate[gid]!=2){
    //if(gid>size[0] && indicate[gid]!=2){
    v=inArray[gid].src; 
                while(p[v] != v)
                        {v = p[v];}

                    i = v;
                   v=inArray[gid].dest; 
                while(p[v] != v)
                        {v = p[v];}
                   int  j = v;
                   //if(gid==5){
                  //atomic_xchg(&test[0],i);
                  //atomic_xchg(&test[1],j);
//  }
                        outArray[gid].src=i;
                        outArray[gid].dest=j;
                   if(i==j)
                   {
                        indicate[gid]=2;

           
                   }
}
}

*/

__kernel void test_kruskal(__global int* src,__global uint * p,__global uint* indicate,const uint size,__global int*dest)
{   //int size=2;
    //int siz=size;
    int gid = get_global_id(0);
    int v;
    int i;
    if(gid<size && indicate[gid]!=2){
    //if(gid>size[0] && indicate[gid]!=2){
    //int index_=index[gid];
    v=src[gid]; 

                while(p[v] != v)
                        {v = p[v];}

                    i = v;
                   v=dest[gid]; 
                while(p[v] != v)
                        {v = p[v];}
                   int  j = v;
                   //if(gid==5){
                  //atomic_xchg(&test[0],i);
                  //atomic_xchg(&test[1],j);
//  }
               //         outArray[gid].src=i;
                 //       outArray[gid].dest=j;
                   if(i==j)
                   {
                        indicate[gid]=2;

           
                   }
}
}

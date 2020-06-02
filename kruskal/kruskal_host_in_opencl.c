
#define PROGRAM_FILE "kernel.cl"
#pragma warning(disable: 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include<math.h>
#include<time.h>
#ifdef MAC
#include <OpenCL/cl.h>
#else  
#include <CL/cl.h>
#endif
#define MERGESORT_SMALL_STRIDE 1024 * 64


typedef struct Edge
{
    int src, dest, weight;
}Edge;
void merge(struct Edge* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    struct Edge* L = (struct Edge*)malloc(sizeof(struct Edge) * n1);
    struct Edge* R = (struct Edge*)malloc(sizeof(struct Edge) * n2);
         

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray 
    j = 0; // Initial index of second subarray 
    k = l; // Initial index of merged subarray 
    while (i < n1 && j < n2)
    {
        if (L[i].weight <= R[j].weight)
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(struct Edge* arr, int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for 
        // large l and h 
        int m = l + (r - l) / 2;

        // Sort first and second halves 
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}
// a structure to represent a connected, undirected 
// and weighted graph 
typedef struct Graph
{
    // V-> Number of vertices, E-> Number of edges 
    int V, E;

    // graph is represented as an array of edges. 
    // Since the graph is undirected, the edge 
    // from src to dest is also edge from dest 
    // to src. Both are counted as 1 edge here. 
    struct Edge* edge;
}Graph;

// Creates a graph with V vertices and E edges 
struct Graph* createGraph(int V, int E)
{
    struct Graph* graph = malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;

    graph->edge = malloc(sizeof(Edge) * E);

    return graph;
}

// A structure to represent a subset for union-find 
struct subset
{
    int parent;
    int rank;
};

// A utility function to find set of an element i 
// (uses path compression technique) 
int find(struct subset *subsets, int i)
{
    // find root and make root as parent of i 
    // (path compression) 
    //printf("%d in \n ", i);
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);

    return subsets[i].parent;
}

int paralle_find(int *p, int i)
{
    // find root and make root as parent of i 
    // (path compression) 
    //printf("%d in \n ", i);
    if (p[i] != i)
        p[i] = paralle_find(p, p[i]);

    return p[i];
}
// A function that does union of two sets of x and y 
// (uses union by rank) 
void Union(struct subset subsets[], int x, int y)
{
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    // Attach smaller rank tree under root of high 
    // rank tree (Union by Rank) 
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;

    // If ranks are same, then make one as root and 
    // increment its rank by one 
    else
    {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

// Compare two edges according to their weights. 
// Used in qsort() for sorting an array of edges 
int myComp(const void* a, const void* b)
{
    struct Edge* a1 = (struct Edge*)a;
    struct Edge* b1 = (struct Edge*)b;
    return a1->weight > b1->weight;
}

// The main function to construct MST using Kruskal's algorithm 
void KruskalMST(struct Graph* graph)
{
    int V = graph->V;
    struct Edge *result=malloc(sizeof(struct Edge)*(graph->E+1)); // Tnis will store the resultant MST 
    int e = 0; // An index variable, used for result[] 
    int i = 0; // An index variable, used for sorted edges 

    // Step 1: Sort all the edges in non-decreasing 
    // order of their weight. If we are not allowed to 
    // change the given graph, we can create a copy of 
    // array of edges 
    //qsort(graph->edge, graph->E, sizeof(graph->edge[0]), myComp);
    //mergeSort(graph->edge,0, graph->E-1);
    // Allocate memory for creating V ssubsets 
    struct subset* subsets =
        (struct subset*) malloc(V * sizeof(struct subset));
    //printf("%d\n", graph->E);
    //for (int i = 0; i < 65; i++)
    //{
      //  printf("%d %d %d %d iop\n", i, graph->edge[i].weight, graph->edge[i].src, graph->edge[i].dest);
    //}
    // Create V subsets with single elements 
    for (int v = 0; v < V; ++v)
    {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }

    // Number of edges to be taken is equal to V-1 
    while (e < V - 1 && i < graph->E)
    {
        //printf("%d %d %d %d weight\n",i, graph->edge[i].weight, graph->edge[i].src, graph->edge[i].dest);
        // Step 2: Pick the smallest edge. And increment 
        // the index for next iteration 
        struct Edge next_edge = graph->edge[i++];

        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);

        // If including this edge does't cause cycle, 
        // include it in result and increment the index 
        // of result for next edge 
        if (x != y)
        {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
        // Else discard the next_edge 
    }

    // print the contents of result[] to display the 
    // built MST 
    FILE* ptr;
    ptr = fopen("sequential.txt", "w");
    
    printf("Following are the edges in the constructed MST\n");
    for (i = 0; i < e; ++i)
    {
        //printf("%d -- %d == %d\n", result[i].src, result[i].dest,  result[i].weight);
        fprintf(ptr, "%d %d %d\n", result[i].src, result[i].dest,
            result[i].weight);

    }
    fclose(ptr);
    return;
}

typedef struct node
{
    cl_uint src;
    cl_uint dest;
    cl_uint weight;
   
}node;
size_t getPaddedSize(size_t n)
{
    unsigned int log2val = (unsigned int)ceil(log((float)n) / log(2.f));
    return (size_t)pow(2, log2val);
}

size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize)
{
    size_t r = DataElemCount % LocalWorkSize;
    if (r == 0)
        return DataElemCount;
    else
        return DataElemCount + LocalWorkSize - r;
}
void parallel_merge(struct node* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    struct node* L = (struct node*)malloc(sizeof(struct node) * n1);
    struct node* R = (struct node*)malloc(sizeof(struct node) * n2);


    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray 
    j = 0; // Initial index of second subarray 
    k = l; // Initial index of merged subarray 
    while (i < n1 && j < n2)
    {
        if (L[i].weight > R[j].weight)
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void parallel_mergeSort(struct node* arr, int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for 
        // large l and h 
        int m = l + (r - l) / 2;

        // Sort first and second halves 
        parallel_mergeSort(arr, l, m);
        parallel_mergeSort(arr, m + 1, r);

        parallel_merge(arr, l, m, r);
    }
}
void parallel_merge_without_node(int* weight,int* src,int* dest, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    int* L = (int*)malloc(sizeof(int) * n1);
    int* R = (int*)malloc(sizeof(int) * n2);

   int* srcL = (int*)malloc(sizeof(int) * n1);
    int* srcR = (int*)malloc(sizeof(int) * n2);

   int* destL = (int*)malloc(sizeof(int) * n1);
    int* destR = (int*)malloc(sizeof(int) * n2);

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        {L[i] = weight[l + i];
            //printf("%d \n",src[l+i]);
            srcL[i]=src[l+i];
            destL[i]=dest[l+i];
        }
    for (j = 0; j < n2; j++)
        {R[j] = weight[m + 1 + j];
            srcR[j]=src[m + 1 + j];
            destR[j]=dest[m + 1 + j];
}
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray 
    j = 0; // Initial index of second subarray 
    k = l; // Initial index of merged subarray 
    while (i < n1 && j < n2)
    {
        if (L[i] > R[j])
        {
            weight[k] = L[i];
            src[k]=srcL[i];
            dest[k]=destL[i];
            i++;
        }
        else
        {
            weight[k] = R[j];
            src[k]=srcR[j];
            dest[k]=destR[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1)
    {
        weight[k] = L[i];
        src[k]=srcL[i];
        dest[k]=destL[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2)
    {
        weight[k] = R[j];
        src[k]=srcR[j];
        dest[k]=destR[j];
        j++;
        k++;
    }
    free(L);
    free(R);
    free(srcL);
    free(destL);
    free(srcR);
    free(destR);
}

/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void parallel_mergeSort_without_node(int* weight,int* src,int* dest, int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for 
        // large l and h 
        int m = l + (r - l) / 2;

        // Sort first and second halves 
        parallel_mergeSort_without_node(weight,src,dest, l, m);
        parallel_mergeSort_without_node(weight,src,dest, m + 1, r);

        parallel_merge_without_node(weight,src,dest, l, m, r);
    }
}


int main() {

    /* Host/device data structures */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int i, err;

    /* Program/kernel data structures */
    cl_program program;
    FILE* program_handle;
    char* program_buffer, * program_log;
    size_t program_size, log_size;
    cl_kernel kernel;

    /* Data and buffers */
    float mat[16], vec[4], result[4];
    float correct[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    cl_mem mat_buff, vec_buff, res_buff;
    size_t work_units_per_kernel;

    cl_mem            m_dPingArray;
    cl_mem            m_dPongArray;
    cl_mem            m_input_index;
    cl_mem            m_output_index;
    

    cl_mem            m_po;
    cl_mem            m_indicate;
    cl_mem            m_test;
    cl_mem            m_flag;
    cl_mem            m_size;
    cl_mem            sub_buffer;
  
    cl_kernel         kruskal_algo;
    cl_kernel         test_kruskal;
    size_t            m_N;
    size_t            m_N_padded;
  

    // input data
    //unsigned int* m_hInput;
    // results
    //unsigned int* m_resultCPU;
    //unsigned int* m_resultGPU[3];

    size_t LocalWorkSize[3] = { 256, 1, 1 };
    unsigned int arraySize = 4 * 4;

    m_N = arraySize;
    //m_N_padded = abs(getPaddedSize(m_N));
    m_N_padded = 4194304;
    printf("%d as\n", abs(m_N_padded));
    node* m_hInput = malloc(sizeof(node) * m_N_padded);
    
    int *input_weight=malloc(sizeof(int)*m_N_padded);
    int *input_src=malloc(sizeof(int)*m_N_padded);
    int *input_dest=malloc(sizeof(int)*m_N_padded);
    int *input_index=malloc(sizeof(int)*m_N_padded);



    int *in_weight=malloc(sizeof(int)*m_N_padded);
    int *in_src=malloc(sizeof(int)*m_N_padded);
    int *in_dest=malloc(sizeof(int)*m_N_padded);
    //int *in_index=malloc(sizeof(int)*m_N_padded);

       // node* parallel_test = malloc(sizeof(node) * m_N_padded);
    node* m_resultCPU = malloc(sizeof(node) * m_N_padded);
    node* m_resultGPU[3];
   
    int V =2900; // Number of vertices in graph 
    int E = m_N_padded; // Number of edges in graph 
    struct Graph* sequential_input = createGraph(V,m_N_padded );

    int temp=0;
    int input_temp=0;\

    //create random input graph.
    srand(time(0));
    for (cl_uint k = 0;k <2901 ; k++)
    {
        for (cl_uint j = 0;j < k; j++)
        {   
            sequential_input->edge[temp].src = k;
            sequential_input->edge[temp].dest = j;

          
            in_src[input_temp]=k;
            in_dest[input_temp]=j;
            

          
            in_weight[input_temp] = rand() % 70+ rand() % 20;

             if (in_weight[input_temp]> 60)in_weight[input_temp] = 999;

           sequential_input->edge[temp].weight = in_weight[input_temp];
            

           temp++;
           input_temp++;

           if (temp == m_N_padded || temp== m_N_padded)
               break;
        }
        if (temp == m_N_padded)
            break;
        sequential_input->edge[temp].src = k;
        sequential_input->edge[temp].dest = k;
        sequential_input->edge[temp].weight = 999;



        in_src[input_temp]=k;
        in_dest[input_temp]=k;
            
        
        in_weight[input_temp]=999;


        input_temp++;
        temp++;
        if (temp == m_N_padded)
            break;
    }
    //printf("%d %d hy\n", temp, in_src[0]);
    m_N_padded = temp;

   double time_spent = 0.0;
    clock_t begin = clock();    
 mergeSort(sequential_input->edge, 0, sequential_input->E - 1);
    KruskalMST(sequential_input);
    clock_t end = clock();
    time_spent = time_spent + (double)(end - begin) / CLOCKS_PER_SEC;
    printf("sequrntial total time %f\n", time_spent);
    time_spent = 0.0;
    

    printf("hkdvkh\n");
    
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't find any platforms");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0) {
        perror("Couldn't find any devices");
        exit(1);
    }

    /* Create the context */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    /* Read program file and place content into buffer */
    program_handle = fopen(PROGRAM_FILE, "rb");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(context, 1,
        (const char**)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    cl_int clError, clError2;
    m_dPingArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m_N_padded, NULL, &clError2);
    clError = clError2;
    m_dPongArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m_N_padded, NULL, &clError2);
    clError |= clError2;
        m_input_index = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m_N_padded, NULL, &clError2);
    clError = clError2;
    m_output_index = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m_N_padded, NULL, &clError2);
    clError |= clError2;

    m_po = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m_N_padded, NULL, &clError2);
    clError |= clError2;
    m_indicate = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N_padded, NULL, &clError2);
    clError |= clError2;
    m_test = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N_padded, NULL, &clError2);
    clError |= clError2;
    m_flag = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(cl_uint), NULL, &clError2);
    clError |= clError2;
    m_size = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(cl_uint), NULL, &clError2);
    clError |= clError2;

    if (clError < 0) {
        printf("pm %d", clError);
        exit(1);
    }

    
    test_kruskal = clCreateKernel(program, "test_kruskal", &clError);
    if (clError < 0) {
        printf("qweer %d", clError);
        exit(1);
    }
/*m_MergesortGlobalSmallKernel = clCreateKernel(program, "Sort_MergesortGlobalSmall", &clError);
 
    clError2 = clError;
    m_MergesortGlobalBigKernel = clCreateKernel(program, "Sort_MergesortGlobalBig", &clError);
    clError2 |= clError;
 
    m_MergesortStartKernel = clCreateKernel(program, "Sort_MergesortStart", &clError);
    clError2 |= clError;
    if (clError2 < 0) {
        printf("pm bn2 %d", clError2);
        exit(1);
    }*/
    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    localWorkSize[0] = LocalWorkSize[0];
    globalWorkSize[0] = GetGlobalWorkSize(m_N_padded / 2, localWorkSize[0]);

    unsigned int limit = (unsigned int)2 * LocalWorkSize[0]; //limit is double the localWorkSize


    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        perror("Couldn't create the command queue");
        exit(1);
    }
size_t local_size[3]={256,1,1};
 //    m_resultGPU[1] = (node*)malloc(m_N_padded * sizeof(node));
 

    cl_uint* p = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* indicate = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* result_indicate = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* test = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* flag = malloc(sizeof(cl_uint) * m_N_padded);
 
    flag[0] = 0;
    flag[1] = 0;
    flag[2] = 1;
    flag[3] = 1;
    flag[4] = 1;
    int * src_input=(int *)malloc(sizeof(int)*m_N_padded);
    int * dest_input=(int *)malloc(sizeof(int)*m_N_padded);
  
for (int i = 0; i < m_N_padded; i++)
    {
        p[i] = i;
        indicate[i] = 0;
  
    }
    globalWorkSize[0] = m_N_padded;

    cl_mem m_temp_input;
    m_temp_input = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int) * m_N_padded, NULL, &clError2);
    if (clError2 < 0) {
        printf("pm %d", clError);
        exit(1);
    }

  
    clError = clSetKernelArg(test_kruskal, 0, sizeof(cl_mem), (void*)&m_temp_input);
    
    clError |= clSetKernelArg(test_kruskal, 1, sizeof(cl_mem), (void*)&m_po);
clError |= clSetKernelArg(test_kruskal, 2, sizeof(cl_mem), (void*)&m_indicate);
  
            clError |= clSetKernelArg(test_kruskal, 4, sizeof(cl_mem), (void*)&m_dPingArray);
  
    if (clError < 0) {
        printf("op %d", clError);
        exit(1);
    }
  
FILE* ptr;
     ptr = fopen("parallel.txt", "w");

    begin = clock();
  //sort the edge of the graph .
     parallel_mergeSort_without_node(in_weight,in_src,in_dest,0,input_temp-1);
    int v, j, ml;
    int temp_var = 0;
    globalWorkSize[0] = m_N_padded;
  
    int o = m_N_padded - 1;
    int edg = 0;
  
    clEnqueueWriteBuffer(queue, m_temp_input, CL_FALSE, 0, o* sizeof(int), in_src, 0, NULL, NULL);
  
                          clEnqueueWriteBuffer(queue, m_dPingArray, CL_FALSE, 0, o* sizeof(int), in_dest, 0, NULL, NULL);
                    clEnqueueWriteBuffer(queue, m_indicate, CL_FALSE, 0, o * sizeof(cl_uint), indicate, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, m_po, CL_FALSE, 0, V * sizeof(cl_uint), p, 0, NULL, NULL);
        while(o>=0 && edg<V-1)
  
    {
        
    //this if condition check for the current edge that wheather it is already discarded by any parallel thread if not then add that to the graph.
        if (indicate[o] != 2)
        {
            //sorce parent
            ml=in_src[o];
            //to find the root of the sorce
            ml=paralle_find(p, ml);
            //destination parent
           j=in_dest[o];
           // j = inter_m[o].dest;
          //to find the root of the destination
            j= paralle_find(p, j);
        //if root of src and dest is not same then add that to the graph.
            if (ml != j)
            {
                 indicate[o] = 1;
                edg++;
                temp_var++;
                if (j > ml)
                    p[j] = ml;
                else
                    p[ml] = j;
               fprintf(ptr, "%d %d %d\n", in_src[o], in_dest[o], in_weight[o]);
            }
            else
            {
                indicate[o] = 2;
            }
        }
            if(temp_var%512==0 && o > 256)
                   {

  
                   clError |= clSetKernelArg(test_kruskal, 3, sizeof(cl_uint), (void*)&o);
  
                    if (clError < 0)
                    {printf("%d rt\n",clError);
                        perror("Couldn't enqueue the kernel execution command mike");
                        exit(1);
                    }

                    LocalWorkSize[0] = 256;
                    localWorkSize[0] = LocalWorkSize[0];
                    globalWorkSize[0] = GetGlobalWorkSize(o, localWorkSize[0]);
                    //localWorkSize[0] = LocalWorkSize[0];
                   //call to the parallel function to find the remove the edge which form the cycle in the graph.
                    clError = clEnqueueNDRangeKernel(queue, test_kruskal, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                    if (clError < 0)
                    {printf("%d rt\n",clError);
                        perror("Couldn't enqueue the kernel execution command ty");
                        exit(1);
                    }
  
                clError = clEnqueueReadBuffer(queue, m_indicate, CL_TRUE, 0, o * sizeof(cl_uint), indicate, 0, NULL, NULL);
                    if (clError < 0)
                {printf("%d rt\n",clError);
                    perror("Couldn't enqueue the kernel execution command tyu");
                    exit(1);
                }
             }
            
            o--;
            if(in_weight[o] == 999 )
                break;
    }
end = clock();
  
    fclose(ptr);
     //end = clock();
    time_spent = 0.0;
    time_spent = time_spent + (double)(end - begin) / CLOCKS_PER_SEC;
    printf("parallel total time %f for %d size edge \n", time_spent,m_N_padded);
    clReleaseMemObject(m_dPingArray);
    clReleaseMemObject(m_dPongArray);
    clReleaseMemObject(m_po);
    clReleaseMemObject(m_indicate);
    clReleaseMemObject(m_test);
    clReleaseKernel(test_kruskal);
   // clReleaseKernel(m_MergesortStartKernel);
    //clReleaseKernel(m_MergesortGlobalBigKernel);
    //clReleaseKernel(m_MergesortGlobalSmallKernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
}




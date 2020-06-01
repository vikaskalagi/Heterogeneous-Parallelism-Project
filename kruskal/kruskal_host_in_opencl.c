
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

/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
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

void Sort_Mergesort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3],int *m_hInput,int *input_index,cl_kernel m_MergesortStartKernel, cl_kernel  m_MergesortGlobalSmallKernel, cl_kernel m_MergesortGlobalBigKernel, cl_mem  m_dPingArray, cl_mem m_dPongArray,cl_mem m_input_index,cl_mem m_output_index, size_t m_N_padded)
{
    //TODO fix memory problem when many elements. -> CL_OUT_OF_RESOURCES
    cl_int clError;
    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    localWorkSize[0] = LocalWorkSize[0];
    globalWorkSize[0] = GetGlobalWorkSize(m_N_padded / 2, localWorkSize[0]);
    unsigned int locLimit = 1;
    clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(int), m_hInput, 0, NULL, NULL);
       clEnqueueWriteBuffer(CommandQueue, m_input_index, CL_FALSE, 0, m_N_padded * sizeof(int), input_index, 0, NULL, NULL);
    if (m_N_padded >= LocalWorkSize[0] * 2) {
        locLimit = 2 * LocalWorkSize[0];

        // start with a local variant first, ASSUMING we have more than localWorkSize[0] * 2 elements
        clError = clSetKernelArg(m_MergesortStartKernel, 0, sizeof(cl_mem), (void*)&m_dPingArray);
        clError |= clSetKernelArg(m_MergesortStartKernel, 1, sizeof(cl_mem), (void*)&m_dPongArray);
        clError |= clSetKernelArg(m_MergesortStartKernel, 2, sizeof(cl_mem), (void*)&m_input_index);
        clError |= clSetKernelArg(m_MergesortStartKernel, 3, sizeof(cl_mem), (void*)&m_output_index);
        //V_RETURN_CL(clError, "Failed to set kernel args: MergeSortStart");

        clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortStartKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        //V_RETURN_CL(clError, "Error executing MergeSortStart kernel!");
        //clEnqueueReadBuffer(CommandQueue, m_dPongArray, CL_TRUE, 0, m_N_padded * sizeof(node), m_hInput, 0, NULL, NULL);
         if (clError < 0)
                    {   
                        perror("Couldn't enqueue the kernel execution command tzy ");
                        exit(1);
                    }
                cl_mem temp_to_swap=m_dPingArray;
        m_dPingArray=m_dPongArray;
        m_dPongArray=temp_to_swap;

        temp_to_swap=m_input_index;
        m_input_index=m_output_index;
        m_output_index=temp_to_swap;

//        swap(m_dPingArray, m_dPongArray);
    }
    //clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(node), m_hInput, 0, NULL, NULL);
    // proceed with the global variant
    unsigned int stride = 2 * locLimit;

    localWorkSize[0] = LocalWorkSize[0];
    globalWorkSize[0] = GetGlobalWorkSize(m_N_padded / 2, localWorkSize[0]);

    if (m_N_padded <= MERGESORT_SMALL_STRIDE) {
        // set not changing arguments
        clError = clSetKernelArg(m_MergesortGlobalSmallKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
        //V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");

        for (; stride <= m_N_padded; stride <<= 1) {
            //calculate work sizes
            size_t neededWorkers = m_N_padded / stride;
            //clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(node), m_hInput, 0, NULL, NULL);
            localWorkSize[0] = (LocalWorkSize[0]< neededWorkers)?LocalWorkSize[0]: neededWorkers;
            globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

            clError = clSetKernelArg(m_MergesortGlobalSmallKernel, 0, sizeof(cl_mem), (void*)&m_dPingArray);
            clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 1, sizeof(cl_mem), (void*)&m_dPongArray);
            clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 2, sizeof(cl_uint), (void*)&stride);
                    clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 4, sizeof(cl_mem), (void*)&m_input_index);
        clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 5, sizeof(cl_mem), (void*)&m_output_index);
            //V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");
if (clError < 0)
                    {
                        perror("Couldn't enqueue the kernel execution command pty");
                        exit(1);
                    }
            clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalSmallKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
            if (clError < 0)
                    {
                        perror("Couldn't enqueue the kernel execution command tjy");
                        exit(1);
                    }
            //V_RETURN_CL(clError, "Error executing kernel!");
        cl_mem temp_to_swap=m_dPingArray;
        m_dPingArray=m_dPongArray;
        m_dPongArray=temp_to_swap;


        temp_to_swap=m_input_index;
        m_input_index=m_output_index;
        m_output_index=temp_to_swap;

        }
    }
    else {
        // set not changing arguments
        clError = clSetKernelArg(m_MergesortGlobalBigKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
        //V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");

        for (; stride <= m_N_padded; stride <<= 1) {
            //calculate work sizes
            size_t neededWorkers = m_N_padded / stride;
            //clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(node), m_hInput, 0, NULL, NULL);
            localWorkSize[0] = (LocalWorkSize[0]< neededWorkers)?LocalWorkSize[0]: neededWorkers;
            globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

            clError = clSetKernelArg(m_MergesortGlobalBigKernel, 0, sizeof(cl_mem), (void*)&m_dPingArray);
            clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 1, sizeof(cl_mem), (void*)&m_dPongArray);
            clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 2, sizeof(cl_uint), (void*)&stride);
            clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 4, sizeof(cl_mem), (void*)&m_input_index);
            clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 5, sizeof(cl_mem), (void*)&m_output_index);
            //V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");
if (clError < 0)
                    {printf("%d tyyuu\n",clError);
                        perror("Couldn't enqueue the kernel execution command topy");
                        exit(1);
                    }
            clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalBigKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
            //V_RETURN_CL(clError, "Error executing kernel!");
if (clError < 0)
                    {
                        perror("Couldn't enqueue the kernel execution command tbny");
                        exit(1);
                    }
            //if (stride >= 1024 * 1024) V_RETURN_CL(clFinish(CommandQueue), "Failed finish CommandQueue at mergesort for bigger strides.");
                    cl_mem temp_to_swap=m_dPingArray;
        m_dPingArray=m_dPongArray;
        m_dPongArray=temp_to_swap;

        temp_to_swap=m_input_index;
        m_input_index=m_output_index;
        m_output_index=temp_to_swap;
        }
    }
    clEnqueueReadBuffer(CommandQueue, m_dPingArray, CL_TRUE, 0, m_N_padded * sizeof(int), m_hInput, 0, NULL, NULL);
        clEnqueueReadBuffer(CommandQueue, m_input_index, CL_TRUE, 0, m_N_padded * sizeof(int), input_index, 0, NULL, NULL);
    //return m_hInput;
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
    cl_kernel         m_BitonicStartKernel;
    cl_kernel         m_BitonicGlobalKernel;
    cl_kernel         m_BitonicLocalKernel;
    cl_kernel         kruskal_algo;
    cl_kernel         test_kruskal;
    size_t            m_N;
    size_t            m_N_padded;
    cl_kernel           m_MergesortStartKernel;
    cl_kernel           m_MergesortGlobalSmallKernel;
    cl_kernel           m_MergesortGlobalBigKernel;

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
    int input_temp=0;
    srand(time(0));
    for (cl_uint k = 0;k <2901 ; k++)
    {
        for (cl_uint j = 0;j < k; j++)
        {   
            sequential_input->edge[temp].src = k;
            sequential_input->edge[temp].dest = j;

            //m_hInput[temp].src = k;
            //m_hInput[temp].dest = j;

            //input_src[input_temp]=k;
            //input_dest[input_temp]=j;
            
            //input_index[input_temp]=input_temp;
          
            in_src[input_temp]=k;
            in_dest[input_temp]=j;
            
            //in_index[input_temp]=input_temp;
          
            in_weight[input_temp] = rand() % 70+ rand() % 20;

//            m_hInput[temp].weight = rand() % 70+ rand() % 20;//+rand() % 70;

//           if (m_hInput[temp].weight > 60)m_hInput[temp].weight = 999;

             if (in_weight[input_temp]> 60)in_weight[input_temp] = 999;

           sequential_input->edge[temp].weight = in_weight[input_temp];
            
//            input_weight[input_temp]=m_hInput[temp].weight;
  //          in_weight[input_temp]=m_hInput[temp].weight;
           temp++;
           input_temp++;
           //printf("%d \n", temp);
           if (temp == m_N_padded || temp== m_N_padded)
               break;
        }
        if (temp == m_N_padded)
            break;
        sequential_input->edge[temp].src = k;
        sequential_input->edge[temp].dest = k;
        sequential_input->edge[temp].weight = 999;

       // m_hInput[temp].src = k;
        //m_hInput[temp].dest = k;

        //m_hInput[temp].weight = 999;

        // input_src[input_temp]=k;
        // input_dest[input_temp]=k;
            
        // input_index[input_temp]=input_temp;
        // input_weight[input_temp]=999;


        in_src[input_temp]=k;
        in_dest[input_temp]=k;
            
        
        in_weight[input_temp]=999;


        input_temp++;
        temp++;
        if (temp == m_N_padded)
            break;
    }
    printf("%d %d hy\n", temp, in_src[0]);
    m_N_padded = temp;

  //  memcpy(in_weight,input_weight,sizeof(int)*input_temp);
    //    memcpy(in_src,input_src,sizeof(int)*input_temp);
      //      memcpy(in_dest,input_dest,sizeof(int)*input_temp);
                //memcpy(in_weight,input_src,sizeof(int)*input_temp);
    
//    mergeSort(sequential_input->edge, 0, sequential_input->E - 1);
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
m_MergesortGlobalSmallKernel = clCreateKernel(program, "Sort_MergesortGlobalSmall", &clError);
    //V_RETURN_FALSE_CL(clError, "Failed to create kernel: Sort_MergesortGlobalSmall.");
    clError2 = clError;
    m_MergesortGlobalBigKernel = clCreateKernel(program, "Sort_MergesortGlobalBig", &clError);
    clError2 |= clError;
    //V_RETURN_FALSE_CL(clError, "Failed to create kernel: Sort_MergesortGlobalBig.");
    m_MergesortStartKernel = clCreateKernel(program, "Sort_MergesortStart", &clError);
    clError2 |= clError;
    if (clError2 < 0) {
        printf("pm bn2 %d", clError2);
        exit(1);
    }
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
 //parallel_mergeSort(parallel_test, 0, temp - 1);
    // int jklp=1;
    /*for (int gh=0;gh<temp;gh++)
    {
        if(m_hInput[gh].weight!=parallel_test[gh].weight || m_hInput[gh].src!=parallel_test[gh].src || m_hInput[gh].dest!=parallel_test[gh].dest)
            {jklp=0;
                printf("sort not same\n\n");
                break;
            }
    }
    if(jklp)
    {
        printf("sort same\n\n");
    }*/

    m_resultGPU[1] = (node*)malloc(m_N_padded * sizeof(node));
    //m_resultGPU[1] = m_hInput;

    cl_uint* p = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* indicate = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* result_indicate = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* test = malloc(sizeof(cl_uint) * m_N_padded);
    cl_uint* flag = malloc(sizeof(cl_uint) * m_N_padded);
    //node* inter_m = malloc(sizeof(node) * m_N_padded);
    //inter_m[m_N_padded - 1].src = m_resultGPU[1][m_N_padded-1].src;
    //inter_m[m_N_padded - 1].dest = m_resultGPU[1][m_N_padded - 1].dest;
    flag[0] = 0;
    flag[1] = 0;
    flag[2] = 1;
    flag[3] = 1;
    flag[4] = 1;
    int * src_input=(int *)malloc(sizeof(int)*m_N_padded);
    int * dest_input=(int *)malloc(sizeof(int)*m_N_padded);
    // for (int i = 0; i < m_N_padded; i++)
    // {
    //     p[i] = i;
    //     indicate[i] = 0;
    //     src_input[i]=input_src[input_index[i]];
    //     dest_input[i]=input_dest[input_index[i]];
    //     //inter_m[i].src = m_resultGPU[1][i].src;
    //     //inter_m[i].dest = m_resultGPU[1][i].dest;
    // }
    //printf("\n %d:%d %d:%d %d:%d in\n",m_hInput[4194303].src,input_src[4194303],m_hInput[4194303].dest,input_dest[4194303],m_hInput[4194303].weight,input_weight[4194303]);
// int tesy_inpuwt=0;
//  for ( int u = 0; u <input_temp; u++) {
//         //printf("%d \n ", u);
//         if(m_hInput[u].src!=input_src[input_index[u]] || m_hInput[u].dest!=input_dest[input_index[u]] || m_hInput[u].weight!=input_weight[u])
//             {
//                 tesy_inpuwt=1;
//                 printf("\n %d:%d %d:%d %d:%d\n",m_hInput[u].src,input_index[u],m_hInput[u].dest,input_dest[input_index[u]],m_hInput[u].weight,input_weight[u]);
//                 printf("not same %d \n",u);
//                 break;
//             }
//    }
//    if(tesy_inpuwt==0)
//     printf("same \n");
   //  begin = clock();
   //parallel_mergeSort(m_hInput, 0, temp - 1);
//    for (int i=0;i<input_temp;i++)
//    {   if(in_weight[i]!=input_weight[i] || in_src[i]!=input_src[input_index[i]] || in_dest[i]!=input_dest[input_index[i]])
//     {    //printf("not  sorted");
// printf("\n %d:%d %d:%d %d:%d %d \n",in_weight[i],input_weight[i] , in_src[i],input_src[input_index[i]],in_dest[i],input_dest[input_index[i]],i);
//              exit(0);
//     }
//printf("\n %d:%d %d:%d %d:%d %d \n",in_weight[i],input_weight[i] , in_src[i],input_src[input_index[i]],in_dest[i],input_dest[input_index[i]],i);
   //}
  //parallel_mergeSort_without_node(in_weight,in_src,in_dest,0,input_temp-1);
  //  Sort_Mergesort(context, queue, local_size, m_hInput, m_MergesortStartKernel, m_MergesortGlobalSmallKernel, m_MergesortGlobalBigKernel, m_dPingArray, m_dPongArray, temp);
//Sort_Mergesort(context, queue, local_size, input_weight,input_index, m_MergesortStartKernel, m_MergesortGlobalSmallKernel, m_MergesortGlobalBigKernel, m_dPingArray, m_dPongArray,m_input_index,m_output_index, input_temp);
// int tesy_input=0;
//  for ( int u = 0; u <input_temp; u++) {
//         //printf("%d \n ", u);
//     //printf("%d %d %d op\n",input_weight[u],m_hInput[u].weight,input_index[u]);
//         if(m_hInput[u].src!=input_src[input_index[u]] || m_hInput[u].dest!=input_dest[input_index[u]] || m_hInput[u].weight!=input_weight[u])
//             {
//                 tesy_input=1;
//                 printf("\n %d:%d %d:%d %d:%d %d \n",m_hInput[u].src,input_src[input_index[u]] ,m_hInput[u].dest,input_dest[input_index[u]],m_hInput[u].weight,input_weight[u],input_index[u]);
//                 printf("not same %d \n",u);
//                 break;
//             }
//    }
//    if(tesy_input==0)
//     printf("same \n");
//m_resultGPU[1] = m_hInput;
for (int i = 0; i < m_N_padded; i++)
    {
        p[i] = i;
        indicate[i] = 0;
        //if(i<10)
        //  printf("%d %d %d %d %d \n",in_weight[i],in_src[i],input_src[input_index[i]],in_dest[i],input_dest[input_index[i]] );
        //if(in_weight[i]!=input_weight[i] || in_src[i]!=input_src[input_index[i]] || in_dest[i]!=input_dest[input_index[i]])
  //  { printf("not  sorted");
//printf("\n %d:%d %d:%d %d:%d %d \n",in_weight[i],input_weight[i] , in_src[i],input_src[input_index[i]],in_dest[i],input_dest[input_index[i]],i);
  //        exit(0);
    //}
        //src_input[i]=input_src[input_index[i]];
        //dest_input[i]=input_dest[input_index[i]];
        //src_input[i]=input_src[i];
        //dest_input[i]=input_dest[i];
        //inter_m[i].src = m_resultGPU[1][i].src;
        //inter_m[i].dest = m_resultGPU[1][i].dest;
    }
    globalWorkSize[0] = m_N_padded;

    cl_mem m_temp_input;
    m_temp_input = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int) * m_N_padded, NULL, &clError2);
    if (clError2 < 0) {
        printf("pm %d", clError);
        exit(1);
    }

    //m_dPingArray = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, sizeof(node) * m_N_padded, NULL, &clError2);
    clError = clSetKernelArg(test_kruskal, 0, sizeof(cl_mem), (void*)&m_temp_input);
    
    clError |= clSetKernelArg(test_kruskal, 1, sizeof(cl_mem), (void*)&m_po);
clError |= clSetKernelArg(test_kruskal, 2, sizeof(cl_mem), (void*)&m_indicate);
    //clError |= clSetKernelArg(test_kruskal, 3, sizeof(cl_mem), (void*)&m_test);
 //   clError |= clSetKernelArg(test_kruskal, 4, sizeof(cl_mem), (void*)&m_size);
    //clError |= clSetKernelArg(test_kruskal, 3, sizeof(cl_mem), (void*)&m_dPongArray);
      //  clError |= clSetKernelArg(test_kruskal, 5, sizeof(cl_mem), (void*)&m_input_index);
            clError |= clSetKernelArg(test_kruskal, 4, sizeof(cl_mem), (void*)&m_dPingArray);
            //    clError = clSetKernelArg(test_kruskal, 0, sizeof(cl_mem), (void*)&m_temp_input);
            //clError |= clSetKernelArg(test_kruskal, 2, sizeof(cl_mem), (void*)&m_indicate);
    if (clError < 0) {
        printf("op %d", clError);
        exit(1);
    }
    //cl_buffer_region region;

    //clReleaseCommandQueue(queue);
    //queue = clCreateCommandQueue(context, device, 0, &err);
FILE* ptr;
     ptr = fopen("parallel.txt", "w");

    begin = clock();
    //Sort_Mergesort(context, queue, local_size, input_weight,input_index, m_MergesortStartKernel, m_MergesortGlobalSmallKernel, m_MergesortGlobalBigKernel, m_dPingArray, m_dPongArray,m_input_index,m_output_index, input_temp);
     parallel_mergeSort_without_node(in_weight,in_src,in_dest,0,input_temp-1);
    int v, j, ml;
    int temp_var = 0;
    globalWorkSize[0] = m_N_padded;
    //cl_buffer_region region;
    int o = m_N_padded - 1;
    int edg = 0;
    //int first_time=1;
  //  clEnqueueWriteBuffer(queue, m_dPingArray, CL_FALSE, 0, o * sizeof(node), m_resultGPU[1], 0, NULL, NULL);
    //clEnqueueWriteBuffer(queue, m_temp_input, CL_FALSE, 0, o* sizeof(node), m_resultGPU[1], 0, NULL, NULL);
    //clEnqueueWriteBuffer(queue, m_indicate, CL_FALSE, 0, o * sizeof(cl_uint), indicate, 0, NULL, NULL);
    //for (int o = m_N_padded-1;o >=0 ;o--)
    clEnqueueWriteBuffer(queue, m_temp_input, CL_FALSE, 0, o* sizeof(int), in_src, 0, NULL, NULL);
                         //clEnqueueWriteBuffer(queue, m_input_index, CL_FALSE, 0, m_N_padded* sizeof(int), input_index, 0, NULL, NULL);
                          clEnqueueWriteBuffer(queue, m_dPingArray, CL_FALSE, 0, o* sizeof(int), in_dest, 0, NULL, NULL);
                    clEnqueueWriteBuffer(queue, m_indicate, CL_FALSE, 0, o * sizeof(cl_uint), indicate, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, m_po, CL_FALSE, 0, V * sizeof(cl_uint), p, 0, NULL, NULL);
        while(o>=0 && edg<V-1)
    //for (int o = 0;o < m_N_padded ;o++)
    {
        
      // printf("%d %d %d %d ert \n", p[2],p[0], m_resultGPU[0][o].weight,indicate[5]);
        if (indicate[o] != 2)
        {
            
            ml=in_src[o];
           // ml = inter_m[o].src;
            ml=paralle_find(p, ml);
           j=in_dest[o];
           // j = inter_m[o].dest;
            j= paralle_find(p, j);
        
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

              
                //region.origin = (o + 1) * sizeof(node);
                //region.size = (m_N_padded - o - 1) * sizeof(node);
               //sub_buffer = clCreateSubBuffer(m_dPingArray, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &clError);
                //clEnqueueWriteBuffer(queue, m_dPingArray, CL_FALSE, 0, o * sizeof(node), m_resultGPU[1], 0, NULL, NULL);
               // clEnqueueWriteBuffer(queue, m_temp_input, CL_FALSE, 0, o* sizeof(node), m_resultGPU[1], 0, NULL, NULL);
               // clEnqueueWriteBuffer(queue, m_po, CL_FALSE, 0, V * sizeof(cl_uint), p, 0, NULL, NULL);
                    // if(first_time==1)
                    // {
                        //first_time++ ;
                   //   clEnqueueWriteBuffer(queue, m_temp_input, CL_FALSE, 0, o* sizeof(int), in_src, 0, NULL, NULL);
                   //       //clEnqueueWriteBuffer(queue, m_input_index, CL_FALSE, 0, m_N_padded* sizeof(int), input_index, 0, NULL, NULL);
                   //        clEnqueueWriteBuffer(queue, m_dPingArray, CL_FALSE, 0, o* sizeof(int), in_dest, 0, NULL, NULL);
                    // clEnqueueWriteBuffer(queue, m_indicate, CL_FALSE, 0, o * sizeof(cl_uint), indicate, 0, NULL, NULL);
                    // }
                    // first_time++ ;
                //clEnqueueWriteBuffer(queue, m_indicate, CL_FALSE, 0, o * sizeof(cl_uint), indicate, 0, NULL, NULL);
                //clEnqueueWriteBuffer(queue, m_size, CL_FALSE, 0, sizeof(cl_uint), &o, 0, NULL, NULL);
                //globalWorkSize[0] = m_N_padded;
                   clError |= clSetKernelArg(test_kruskal, 3, sizeof(cl_uint), (void*)&o);
                //if (o >256) {
                    if (clError < 0)
                    {printf("%d rt\n",clError);
                        perror("Couldn't enqueue the kernel execution command mike");
                        exit(1);
                    }

                    LocalWorkSize[0] = 256;
                    localWorkSize[0] = LocalWorkSize[0];
                    globalWorkSize[0] = GetGlobalWorkSize(o, localWorkSize[0]);
                    //localWorkSize[0] = LocalWorkSize[0];
                    clError = clEnqueueNDRangeKernel(queue, test_kruskal, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                    if (clError < 0)
                    {printf("%d rt\n",clError);
                        perror("Couldn't enqueue the kernel execution command ty");
                        exit(1);
                    }
                //}
                //else {
                //    globalWorkSize[0] = o;
                //    clError = clEnqueueNDRangeKernel(queue, test_kruskal, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
                //    if (clError < 0)
                //    {
                //        perror("Couldn't enqueue the kernel execution command");
                //        exit(1);
                //    }
                //}

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
    // FILE* ptr;
    //  ptr = fopen("parallel.txt", "w");
 //   for ( int u = m_N_padded-1; u >=0; u--) {
        //printf("%d \n ", u);
   //     if(indicate[u]==1){
   //     printf(" %d -->   %d  , weight=%d \n", m_resultGPU[1][u].src, m_resultGPU[1][u].dest, m_resultGPU[1][u].weight);
     //   fprintf(ptr, "%d %d %d\n", m_resultGPU[1][u].src, m_resultGPU[1][u].dest, m_resultGPU[1][u].weight);
       // }
    
    
   // }
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
    clReleaseKernel(m_MergesortStartKernel);
    clReleaseKernel(m_MergesortGlobalBigKernel);
    clReleaseKernel(m_MergesortGlobalSmallKernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
}




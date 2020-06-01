#include <stdio.h>
//#include <omp.h>
#include <string.h>
#include <math.h>
#include <sys/resource.h>
//#include "../common/common.h"
#include <cuda_runtime.h>



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
         
    //printf("%d \n",arr[l]);
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
    struct Graph* graph =(struct Graph*) malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;

    graph->edge = (struct Edge*)malloc(sizeof(Edge) * E);

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
    struct Edge *result=(struct Edge*)malloc(sizeof(struct Edge)*(graph->E+1)); // Tnis will store the resultant MST 
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
    int src;
    int dest;
    int weight;
   
}node;

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

__global__ void test_kruskal(int* src,int * p,int* indicate,int size,int *dest)
{
	  int siz=size;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int v;
    int i;
    //int couny=0;
    if(gid<siz && indicate[gid]!=2){
    //if(gid>size[0] && indicate[gid]!=2){
    v=src[gid]; 
                while(p[v] != v)
                        {v = p[v];}

                    i = v;
                   v=dest[gid]; 
                while(p[v] != v)
                        {v = p[v];}
                   int  j = v;

                  // printf("%d %d %d %d rt\n",inArray[gid].src,inArray[gid].dest,gid,size);
                   //if(gid==5){
                  //atomic_xchg(&test[0],i);
                  //atomic_xchg(&test[1],j);
//  }
                      // outArray[gid].src=i;
                       // outArray[gid].dest=j;
                   if(i==j)
                   {
                        indicate[gid]=2;

           
                   }
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

int main(int argc, char *argv[])
{
const rlim_t kStackSize = 1024L * 1024L * 1024L;   // min stack size = 64 Mb
    struct rlimit rl;
    int stack_result;

    stack_result = getrlimit(RLIMIT_STACK, &rl);
    if (stack_result == 0)
    {
        if (rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            stack_result = setrlimit(RLIMIT_STACK, &rl);
            if (stack_result != 0)
            {
                fprintf(stderr, "setrlimit returned result = %d\n", stack_result);
            }
        }
    }

  //  double avg_parallel=0;
//for (int yu=0;yu<10;yu++){
	size_t LocalWorkSize[3] = { 256, 1, 1 };
	long int m_N_padded = 10000000;
    printf("%d as\n", abs(m_N_padded));
    node* m_hInput = (node*)malloc(sizeof(node) * m_N_padded);
       // node* parallel_test = malloc(sizeof(node) * m_N_padded);
   
    int *in_weight=(int *)malloc(sizeof(int)*m_N_padded);
    int *in_src=(int *)malloc(sizeof(int)*m_N_padded);
    int *in_dest=(int *)malloc(sizeof(int)*m_N_padded);
    //int *in_index=malloc(sizeof(int)*m_N_padded);

       // node* parallel_test = malloc(sizeof(node) * m_N_padded);
    //node* m_resultCPU = malloc(sizeof(node) * m_N_padded);
    node* m_resultGPU[3];
   
    int V =4499; // Number of vertices in graph 
    int E = m_N_padded; // Number of edges in graph 
    struct Graph* sequential_input = createGraph(V,m_N_padded );

    int temp=0;
    int input_temp=0;
    srand(time(0));
    for (int k = 0;k <4500; k++)
    {
        for (int j = 0;j < k; j++)
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
//mergeSort(sequential_input->edge, 0, sequential_input->E - 1);
 

//    KruskalMST(sequential_input);

double time_spe= 0.0;
    clock_t beg = clock();    
mergeSort(sequential_input->edge, 0, sequential_input->E - 1);
    KruskalMST(sequential_input);
    clock_t ed = clock();
    time_spe = (double)(ed - beg) / CLOCKS_PER_SEC;
    printf("sequrntial total time %f\n", time_spe);


// m_resultGPU[1] = (node*)malloc(m_N_padded * sizeof(node));
    //m_resultGPU[1] = m_hInput;

    int* p = (int *)malloc(sizeof(int) * m_N_padded);
    int* indicate = (int *)malloc(sizeof(int) * m_N_padded);
    int* result_indicate = (int *)malloc(sizeof(int) * m_N_padded);
    int* test = (int *)malloc(sizeof(int) * m_N_padded);
    int* flag =(int *) malloc(sizeof(int) * m_N_padded);
    //node* inter_m = malloc(sizeof(node) * m_N_padded);
    //inter_m[m_N_padded - 1].src = m_resultGPU[1][m_N_padded-1].src;
    //inter_m[m_N_padded - 1].dest = m_resultGPU[1][m_N_padded - 1].dest;
    flag[0] = 0;
    flag[1] = 0;
    flag[2] = 1;
    flag[3] = 1;
    flag[4] = 1;
    for (int i = 0; i < m_N_padded; i++)
    {
        p[i] = i;
        indicate[i] = 0;
        //inter_m[i].src = m_resultGPU[1][i].src;
        //inter_m[i].dest = m_resultGPU[1][i].dest;
    }

    int  j, ml;
    int temp_var = 0;
    //globalWorkSize[0] = m_N_padded;
    //cl_buffer_region region;
    int o = m_N_padded - 1;
    int edg = 0;
    int first_time=1;

   //  begin = clock();
   //parallel_mergeSort(m_hInput, 0, temp - 1);
   
   //m_resultGPU[1] = m_hInput;

 FILE* ptr;
     ptr = fopen("parallel.txt", "w");
clock_t begin = clock();
parallel_mergeSort_without_node(in_weight,in_src,in_dest,0,input_temp-1);
clock_t end;

   //globalWorkSize[0] = m_N_padded;

	int *m_temp_input;
	int *m_po;
	int *m_indicate;
    int *m_dest;
   cudaMalloc((int **)&m_temp_input, m_N_padded * sizeof(int));
	cudaMalloc((int **)&m_po, m_N_padded * sizeof(int));
	cudaMalloc((int **)&m_indicate, m_N_padded * sizeof(int));
cudaMalloc((int **)&m_dest, m_N_padded * sizeof(int));

cudaMemcpy(m_temp_input, in_src, o* sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(m_dest, in_dest, o* sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(m_po, p, V* sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(m_indicate, indicate, o* sizeof(int), cudaMemcpyHostToDevice);
               

    while(o>=0 && edg<V-1)
    {   if(in_weight[o] == 999)
            break;
        if ( indicate[o] != 2)
        {
            
            ml=in_src[o];
           // ml = inter_m[o].src;
            ml=paralle_find(p, ml);
           //j=m_resultGPU[1][o].dest;
            j = in_dest[o];
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
                    //printf("%d %d %d\n",m_resultGPU[1][o].src, m_resultGPU[1][o].dest, m_resultGPU[1][o].weight);
                    fprintf(ptr, "%d %d %d\n", in_src[o], in_dest[o], in_weight[o]);
            }
            else
            {
                indicate[o] = 2;	
            }
        }
            if(temp_var%512==0 && o > 256 )
                   {
                   //printf("%d \n",o);

                   //if(first_time==1)
                    //{
                        //first_time++ ;
                      //   clEnqueueWriteBuffer(queue, m_temp_input, CL_FALSE, 0, o* sizeof(node), m_resultGPU[1], 0, NULL, NULL);
                    	 
                    //	first_time=1;
                    //}
                    //first_time++ ;
                    //for(int opi=o;opi>o-12;opi--)
                    //{
                    //printf("%d ",indicate[opi]);
                    //}
                    //printf("\n ON \n");

                     
                    LocalWorkSize[0] = 256;
                    //localWorkSize[0] = LocalWorkSize[0];
                    int blocksize = ceil(o/LocalWorkSize[0]);
                    //localWorkSize[0] = LocalWorkSize[0];
                //    clError = clEnqueueNDRangeKernel(queue, test_kruskal, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                    
                    //for(int opi=o;opi>o-12;opi--)
                    //{
                    //printf("%d ",indicate[opi]);
                    //}

                	test_kruskal <<< blocksize, LocalWorkSize[0] >>> (m_temp_input, m_po, m_indicate,o,m_dest);

                	cudaMemcpy(indicate, m_indicate, o * sizeof(int), cudaMemcpyDeviceToHost);
                	//printf("\n in \n");
                	//for(int opi=o;opi>o-12;opi--)
                    //{
                    //printf("%d ",indicate[opi]);
                    //}
                    //printf("\n");
             }
            
            o--;
    }
end = clock();

for ( int u = m_N_padded-1; u >=0; u--) {
        //printf("%d \n ", u);
        if(indicate[u]==1){
        //printf(" %d -->   %d  , weight=%d \n", m_resultGPU[1][u].src, m_resultGPU[1][u].dest, m_resultGPU[1][u].weight);
        //fprintf(ptr, "%d %d %d\n", m_resultGPU[1][u].src, m_resultGPU[1][u].dest, m_resultGPU[1][u].weight);
        }
        }
            fclose(ptr);
	cudaFree(m_temp_input);
	cudaFree(m_po);
	cudaFree(m_indicate);
	
 	//end = clock();
    double time_spent = 0.0;
    time_spent = time_spent + (double)(end - begin) / CLOCKS_PER_SEC;
 printf("parallel total time %f for %d size edge \n", time_spent,m_N_padded);
	//avg_parallel=avg_parallel+time_spent;
	//int pos = rk_matcher(str, pattern, d, q);
	//printf("%d", pos);
	

	//avg_parallel=avg_parallel/10;
 //printf("avg parallel total time %f  \n", avg_parallel);
 
	return 0;
}
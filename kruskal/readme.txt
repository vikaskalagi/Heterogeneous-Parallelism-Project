use visual studio 2019 (with all modules of opencl installed)

if you use linux to run the code use the following command

gcc <filename.c> -lOpenCL -lm
./a.out

and allocate stack reserve size as 80000000

(our code generate some random graph for given number of edges)
if you want to test with other input ,then just do some change in code

make changes in 
{
if you want to assign different edges size (i.e the number of edge that should be included in graph)then 
assign the value to the variable called m_N_padded with your edge number that should be included in the graph
and also assign value to V (vertex in graph by the value of n)
 
int V= number of vertex in the graph for your given edges by calculating the value of (n)
}

to calculate the 
 number of vertex in the graph for your given edges use the below formula

(n+1) n / 2 = number of edge that you have given as input to be considered in graph.

here calculate the value of n and substitute for V.
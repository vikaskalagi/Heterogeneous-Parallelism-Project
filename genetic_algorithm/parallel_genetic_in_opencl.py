from __future__ import division
import random
import pyopencl as cl
import pyopencl.array  
import numpy as np
import pandas as pd
import copy
import time
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)

TPB=0

context = cl.create_some_context()
inputQueue = cl.CommandQueue(context)

kernel=cl.Program(context, """
__kernel void biase_crossover(__global float * rng_states,__global float * out,const float mutation_rate,__global float *mother)
{ 
    int row=get_global_id(1);
    int col=get_global_id(0);
    int rowno=get_global_size(1);
    if (col < get_global_size(0) && row < get_global_size(1))
        {
       // int rand=0.5;
        float rand = rng_states[row + rowno*col];
        //if(rand<0.5)
       if(rand<mutation_rate)
//out[row]=0.5;
            out[row + rowno*col] =mother[row + rowno*col];        
}
}


__kernel void weight_crossover(__global float * rng_states,__global float * out,const float mutation_rate,__global float *mother)
{    int row=get_global_id(1);
    int col=get_global_id(0);
    int rowno=get_global_size(1);
    if (col < get_global_size(0) && row < get_global_size(1))
    {

   //int rand=0.5;
    float rand = rng_states[row + rowno*col ];
 //if(rand<0.5)
    if(rand<mutation_rate)
//out[row]=0.5;
        out[row + rowno*col] =mother[row + rowno*col];      

   
}

}



__kernel void biase_mutation(__global float * rng_states,__global float * out,const float mutation_rate)
{    
    
    int row=get_global_id(1);
    int col=get_global_id(0);

    int rowno=get_global_size(1);    
    if (col<get_global_size(0) && row < get_global_size(1))
       {
     //  int rand=0.5;
    float rand = rng_states[row + rowno*col];
    //if(rand<0.5)
    //out[row + rowno*col]=rand;
    if(rand<mutation_rate)
    //out[row+ rowno*col]=0.5;
        out[row + rowno*col] = out[row + rowno*col]+rand-0.5;    
    //else
      //        out[row + rowno*col] =out[row + rowno*col];
}
 }   



__kernel void weight_mutation(__global float * rng_states,__global float * out,const float mutation_rate)
{
    int row=get_global_id(1);
    int col=get_global_id(0);
   
    int rowno=get_global_size(1);
    if (col < get_global_size(0) && row < get_global_size(1))
{
//int rand=0.5;
    float  rand = rng_states[row + rowno*col];
//if(rand<0.5)
    if(rand<mutation_rate)
//    out[row]=0.5;
      out[row + rowno*col] =out[row + rowno*col]+ rand-0.5;        
}   
}   """).build()

dog=0
def relu(z):
        return np.maximum(z, 0)
# def feedforward(self, a):
#     global TPB
#     global dog
#     cat=0
#     for b, w in zip(self.biases, self.weights):
       
#         if(cat==0):
#             #if(TPB==0):
#             u=np.array(a)
#             A_global_mem = cuda.to_device(w)
#             B_global_mem = cuda.to_device(u)
#             TPB=4
#             threadsperblock = (TPB, TPB)
#             blockspergrid_x = int(math.ceil(w.shape[0] / threadsperblock[1]))
#             blockspergrid_y = int(math.ceil(a.shape[1] / threadsperblock[0]))
#             blockspergrid = (blockspergrid_x, blockspergrid_y)
#             C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
#             fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
#             res = C_global_mem.copy_to_host()
#             #print(res,np.dot(w,a))
#             a=relu(res+b)
#             #a=relu(np.dot(w,a)+b)
#         else:
#             #if(dog<100):
#             dog+=1
#             u=np.array(a)
#             A_global_mem = cuda.to_device(w)
#             B_global_mem = cuda.to_device(u)
#             TPB=4
#             threadsperblock = (TPB, TPB)
#             blockspergrid_x = int(math.ceil(w.shape[0] / threadsperblock[1]))
#             blockspergrid_y = int(math.ceil(a.shape[1] / threadsperblock[0]))
#             blockspergrid = (blockspergrid_x, blockspergrid_y)
#             C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
#             fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
#             res = C_global_mem.copy_to_host()
#             #print(res,np.dot(w,a))
#             a = sigmoid(res+b)
#             #a = sigmoid(np.dot(w,a)+b)
#         cat+=1
#     return a

def feedforward(self, a):
    
    cat=0
    for b, w in zip(self.biases, self.weights):
       
        if(cat==0):
            a=relu(np.dot(w,a)+b)
        else:
            a = sigmoid(np.dot(w,a)+b)
        cat+=1
    return a

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def score(self, X, y):

    total_score=0
    for i in range(X.shape[0]):
        predicted = feedforward(self,X[i].reshape(-1,1))
        actual = y[i].reshape(-1,1)
        #print((np.power(predicted-actual,2)/2))
        total_score += np.sum(np.power(predicted-actual,2)/2)  # mean-squared error
    return total_score

def accuracy(self,X, y):

    #print(X)
    accuracy = 0
    for i in range(X.shape[0]):
        #print(X[i].reshape(-1,1))
        #print(X[i])
        #print()
        output = feedforward(self,X[i].reshape(-1,1))
        accuracy += int(np.argmax(output) == np.argmax(y[i]))
    return accuracy / X.shape[0] * 100


class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1).astype(np.float32) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x).astype(np.float32) for x, y in zip(sizes[:-1], sizes[1:])]
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers-2)])
    
def get_random_point(self, type):

  

    nn = self.nets[0]
    layer_index, point_index = random.randint(0, nn.num_layers-2), 0
    if type == 'weight':
        row = random.randint(0,nn.weights[layer_index].shape[0]-1)
        col = random.randint(0,nn.weights[layer_index].shape[1]-1)
        point_index = (row, col)
    elif type == 'bias':
        point_index = random.randint(0,nn.biases[layer_index].size-1)
    return (layer_index, point_index)

def get_all_scores(self):
    return [score(net,self.X, self.y) for net in self.nets]

def get_all_accuracy(self):
    return [accuracy(net,self.X, self.y) for net in self.nets]

def crossover(self, father, mother):
    nn = copy.deepcopy(father)
    # for _ in range(self.nets[0].bias_nitem):
    #     layer, point = get_random_point(self,'bias')
    #     if random.uniform(0,1) < self.crossover_rate:
    #         nn.biases[layer][point] = mother.biases[layer][point]

    # for _ in range(self.nets[0].weight_nitem):
    #     layer, point = get_random_point(self,'weight')
    #     if random.uniform(0,1) < self.crossover_rate:
    #         nn.weights[layer][point] = mother.weights[layer][point]
    for u in range(nn.num_layers-1):
        #layer, point = get_random_point(self,'weight')
        #if random.uniform(0,1) < self.mutation_rate:
        #    nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
        #TPB=4
        x=nn.biases[u].shape[0]
        y=nn.biases[u].shape[1]


        #C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
        rng_states = np.random.uniform(0, 1,x*y).astype(np.float32)
        #print(nn.biases[u],"initial")

        # inAbuf = cl.array.to_device(inputQueue, rng_states)
        # inBbuf = cl.array.to_device(inputQueue, nn.biases[u])
        # outCbuf = cl.array.to_device(inputQueue, mother.biases[u] )
        mem_flags = cl.mem_flags
        inAbuf=cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=rng_states)
        inBbuf=cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=nn.biases[u])
        outCbuf=cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=mother.biases[u])
        #print(blockspergrid,threadsperblock,rng_states.shape)
        kernel.biase_crossover(inputQueue, nn.biases[u].shape, None,inAbuf ,inBbuf, np.float32(self.crossover_rate), outCbuf)
        cl.enqueue_copy(inputQueue, nn.biases[u], inBbuf)

        #biase_crossover[blockspergrid, threadsperblock](rng_states,nn.biases[u],self.crossover_rate,mother.biases[u])    
        #print(nn.biases[u],"final")
    for u in range(nn.num_layers-1):
        #layer, point = get_random_point(self,'weight')
        #if random.uniform(0,1) < self.mutation_rate:
        #    nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
        x=nn.weights[u].shape[0]
        y=nn.weights[u].shape[1]


        #C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
        rng_states = np.random.uniform(0, 1,x*y).astype(np.float32)
        #print(nn.biases[u],"initial")

        #inAbuf = cl.array.to_device(inputQueue, rng_states)
        #inBbuf = cl.array.to_device(inputQueue, nn.weights[u])
        #outCbuf = cl.array.to_device(inputQueue, mother.weights[u] )
        mem_flags = cl.mem_flags
        inAbuf=cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=rng_states)
        inBbuf=cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=nn.weights[u])
        outCbuf=cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=mother.weights[u])

        #print(nn.weights[u],"initial")
        #print(blockspergrid,threadsperblock)
        #kernel.weight_crossover(inputQueue, nn.weights[u].shape, None,inAbuf.data ,inBbuf.data, np.int32(self.crossover_rate), outCbuf.data)
        kernel.weight_crossover(inputQueue, nn.weights[u].shape, None,inAbuf ,inBbuf, np.float32(self.crossover_rate), outCbuf)
        #weight_crossover[blockspergrid, threadsperblock](rng_states,nn.weights[u],self.crossover_rate,mother.weights[u])
        #print(nn.weights[u],"final")
        #print(inBbuf==nn.weights[u])
        cl.enqueue_copy(inputQueue, nn.weights[u], inBbuf)
        #print(inBbuf.data)
        #nn.weights[u]=inBbuf.data
    return nn
''' 
def mutation(self, child):
    nn = copy.deepcopy(child)
    for _ in range(self.nets[0].bias_nitem):
       
        layer, point = get_random_point(self,'bias')
        if random.uniform(0,1) < self.mutation_rate:
            nn.biases[layer][point] += random.uniform(-0.5, 0.5)

    for _ in range(self.nets[0].weight_nitem):
        layer, point = get_random_point(self,'weight')
        if random.uniform(0,1) < self.mutation_rate:
            nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)

    return nn
'''
def mutation(self, child):
    nn = copy.deepcopy(child)
    for u in range(nn.num_layers-1):
        #layer, point = get_random_point(self,'weight')
        #if random.uniform(0,1) < self.mutation_rate:
        #    nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
        x=nn.biases[u].shape[0]
        y=nn.biases[u].shape[1]


        #C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
        rng_states = np.random.uniform(0, 1,x*y).astype(np.float32)
        #print(nn.biases[u],"initial")

        # inAbuf = cl.array.to_device(inputQueue, rng_states)
        # inBbuf = cl.array.to_device(inputQueue, nn.biases[u])
        temp_to=copy.deepcopy(nn.biases[u])
        
        mem_flags = cl.mem_flags
        inAbuf=cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=rng_states)
        inBbuf=cl.Buffer(context, mem_flags.READ_WRITE| mem_flags.COPY_HOST_PTR , hostbuf=nn.biases[u])
        #print(nn.biases[u],"initial")
        #print(blockspergrid,threadsperblock,rng_states.shape)
        kernel.biase_mutation(inputQueue, nn.biases[u].shape, None,inAbuf ,inBbuf, np.float32(self.mutation_rate))
        cl.enqueue_copy(inputQueue, nn.biases[u], inBbuf)
        #print(rng_states==nn.biases[u],rng_states,nn.biases[u])
        #print(nn.biases[u],"final")
        #biase_mutation[blockspergrid, threadsperblock](rng_states,nn.biases[u],self.mutation_rate)    
        #print(nn.biases[u],"final",temp)
    for u in range(nn.num_layers-1):
        #layer, point = get_random_point(self,'weight')
        #if random.uniform(0,1) < self.mutation_rate:
        #    nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
        x=nn.weights[u].shape[0]
        y=nn.weights[u].shape[1]


        #C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
        rng_states = np.random.uniform(0, 1,x*y).astype(np.float32)
        #print(nn.biases[u],"initial")

        # inAbuf = cl.array.to_device(inputQueue, rng_states)
        # inBbuf = cl.array.to_device(inputQueue, nn.weights[u])
        mem_flags = cl.mem_flags
        inAbuf=cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=rng_states)
        inBbuf=cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=nn.weights[u])
        #print(nn.biases[u],"initial")
        #print(blockspergrid,threadsperblock,rng_states.shape)
        kernel.weight_mutation(inputQueue, nn.weights[u].shape, None,inAbuf ,inBbuf, np.float32(self.mutation_rate))
        cl.enqueue_copy(inputQueue, nn.weights[u], inBbuf)
        #print(nn.weights[u],"initial")
        #print(blockspergrid,threadsperblock)
        #weight_mutation[blockspergrid, threadsperblock](rng_states,nn.weights[u],self.mutation_rate)
        #print(nn.weights[u],"final")
    return nn
def evolve(nnga):
    score_list = list(zip(nnga.nets, get_all_scores(nnga)))
    score_list.sort(key=lambda x: x[1])
    score_list = [obj[0] for obj in score_list]
    retain_num = int(nnga.n_pops*nnga.retain_rate)
    score_list_top = score_list[:retain_num]
    retain_non_best = int((nnga.n_pops-retain_num) * nnga.retain_rate)
    for _ in range(random.randint(0, retain_non_best)):
        score_list_top.append(random.choice(score_list[retain_num:]))

    while len(score_list_top) < nnga.n_pops:

        father = random.choice(score_list_top)
        mother = random.choice(score_list_top)

        if father != mother:
            new_child = crossover(nnga,father, mother)

            new_child = mutation(nnga,new_child)
            score_list_top.append(new_child)
   
    nnga.nets = score_list_top

def hill(self,nn):
    temp=copy.deepcopy(nn)
    final=nn.accuracy(self.X,self.y)
    count=0
    while(count!=1000):
        count+=1
        for i in range(4):  
            temp.biases[0][i] += random.uniform(-0.5, 0.5)*4
            for j in range(24):
                temp.weights[0][i][j] += random.uniform(-0.5, 0.5)*4
        for i in range(2):  
            temp.biases[0][i] += random.uniform(-0.5, 0.5)*2
            for j in range(4):
                temp.weights[1][i][j] += random.uniform(-0.5, 0.5)*2
        tm=temp.accuracy(self.X,self.y)
        if(tm>final):
            nn=copy.deepcopy(temp)
            final=tm
            
        temp=copy.deepcopy(nn)
        
    return nn



class NNGeneticAlgo:

    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, retain_rate, X, y):

        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.retain_rate = retain_rate
        self.X = X[:]
        self.y = y[:]
    
            
def main():

    df = pd.read_csv("Copy of Eyes.csv")
    X = df[eyes]
    y = df['truth_value']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

   
    X=X_train.values
    y=y_train.values
    
    y = y.reshape(-1, 1)
    
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()

   
    N_POPS = 15
    NET_SIZE = [24,100,50,2]
    MUTATION_RATE = 0.4
    CROSSOVER_RATE = 0.6
    RETAIN_RATE = 0.4

   
    nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, X, y)

    start_time = time.time()
   
    i=0
    temp_accuracy=0
    epoch=100
    while(i<epoch):
        i+=1
        evolve(nnga)
    
    #tr=nnga.hill(nnga.nets[np.argmax(nnga.get_all_accuracy())])
    tr=nnga.nets[np.argmax(get_all_accuracy(nnga))]
    #print(tr.accuracy(X,y)," training after")
    X=X_test.values[:]
    y_test=y_test.values
    y_test = y_test.reshape(-1, 1)
    enc.fit(y_test)
    y_test = enc.transform(y_test).toarray()

    y=y_test[:]
    print("for input layer of network =  ",NET_SIZE)
    print(accuracy(tr,X,y),"test accuracy")
    print("Parallel Execution time in seconds =  ",(time.time() - start_time))
if __name__ == "__main__":
    main()
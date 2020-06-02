from __future__ import division
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import random
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

#parallel function for crossover update on biase
@cuda.jit
def biase_crossover(rng_states,out,mutation_rate,mother):
    #find the position of element to operate on
    x=cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y=cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if x >= out.shape[0] and y >= out.shape[1]:
        return
    #generate random number
    rand = xoroshiro128p_uniform_float64(rng_states,x*out.shape[1]+y )
    #temp[x]=rand
    #condition for crossover
    if(rand<mutation_rate):

        out[x][y] =mother[x][y]        

    #out[thread_id] = 4.0 * inside / iterations

#parallel function for crossover update on weight
@cuda.jit
def weight_crossover(rng_states,out,mutation_rate,mother):
    
    #find the position of element to operate on
    x=cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y=cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= out.shape[0] and y >= out.shape[1]:
       
        return
    #generate random number    
    rand = xoroshiro128p_uniform_float64(rng_states,x*out.shape[1]+y )
    #condition for crossover 
    if(rand<mutation_rate):

        out[x][y] =mother[x][y]       

    #out[thread_id] = 4.0 * inside / iterations



#parallel function for mutation update on biase

@cuda.jit
def biase_mutation(rng_states,out,mutation_rate):
    #find the position of element to operate on
    x=cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y=cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= out.shape[0] and y >= out.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return
    #condition for random number.
    rand = xoroshiro128p_uniform_float64(rng_states,x*out.shape[1]+y )
    #temp[x]=rand
    if(rand<mutation_rate):

        out[x][y] =out[x][y]+ rand-0.5        

    #out[thread_id] = 4.0 * inside / iterations

#parallel function for mutation update on weight
@cuda.jit
def weight_mutation(rng_states,out,mutation_rate):
    #find the position of element to operate on
    x=cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y=cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= out.shape[0] and y >= out.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return
    #generate the random number.
    rand = xoroshiro128p_uniform_float64(rng_states,x*out.shape[1]+y )
    if(rand<mutation_rate):

        out[x][y] =out[x][y]+ rand-0.5        

    #out[thread_id] = 4.0 * inside / iterations

#parallel function for matrix multiplication
@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)+1):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp
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

#this function is use to find the output of the network for given input
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

#to get score for each network
def score(self, X, y):

    total_score=0
    for i in range(X.shape[0]):
        predicted = feedforward(self,X[i].reshape(-1,1))
        actual = y[i].reshape(-1,1)
        #print((np.power(predicted-actual,2)/2))
        total_score += np.sum(np.power(predicted-actual,2)/2)  # mean-squared error
    return total_score
#to find the accuracy of the network
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

#network object
class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers-2)])
 
#to generate random numbers for the network
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

#to find all scores for all chromosomes
def get_all_scores(self):
    return [score(net,self.X, self.y) for net in self.nets]
#to find all accuracy for all chromosomes
def get_all_accuracy(self):
    return [accuracy(net,self.X, self.y) for net in self.nets]

#to do crossover for the network
def crossover(self, father, mother):
    nn = copy.deepcopy(father)
    
    #for each biase update for layer 
    for u in range(nn.num_layers-1):
        #number of thread in the block
        TPB=4
        x=nn.biases[u].shape[0]
        y=nn.biases[u].shape[1]
        threadsperblock = [x, y]
        
        if(x>TPB):
            threadsperblock[0]=TPB
            
        if(y>TPB):
            threadsperblock[1]=TPB
        threadsperblock=(threadsperblock[0],threadsperblock[1])
        #number of block for the given input
        blockspergrid_x = int(math.ceil(x / threadsperblock[0]))    
        blockspergrid_y = int(math.ceil(y / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #generate random number object to call in kernel
        rng_states = create_xoroshiro128p_states(x*y, seed=random.randint(0, 10))
        #call to parallel function for crossover on each layer based on the random number.
        biase_crossover[blockspergrid, threadsperblock](rng_states,nn.biases[u],self.crossover_rate,mother.biases[u])    
    #for each weight update for layer        
    for u in range(nn.num_layers-1):
        #number of thread in the block
        TPB=4
        x=nn.weights[u].shape[0]
        y=nn.weights[u].shape[1]
        threadsperblock = [x, y]
        #blockspergrid_x=1
        #blockspergrid_y=1
        if(x>TPB):
            threadsperblock[0]=TPB
            
        if(y>TPB):
            threadsperblock[1]=TPB
        threadsperblock=(threadsperblock[0],threadsperblock[1])
        
        blockspergrid_x = int(math.ceil(x / threadsperblock[0]))    
        blockspergrid_y = int(math.ceil(y / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
        #generate random number object to use in kernel function.
        rng_states = create_xoroshiro128p_states(x*y, seed=random.randint(0, 10))
        #call to parallel function for crossover on each layer based on the random number
        weight_crossover[blockspergrid, threadsperblock](rng_states,nn.weights[u],self.crossover_rate,mother.weights[u])
        #print(nn.weights[u],"final")
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
#to do mutaion for the network
def mutation(self, child):
    nn = copy.deepcopy(child)
    #biase update for each layer
    for u in range(nn.num_layers-1):
        #layer, point = get_random_point(self,'weight')
        #if random.uniform(0,1) < self.mutation_rate:
        #    nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
        TPB=4
        x=nn.biases[u].shape[0]
        y=nn.biases[u].shape[1]
        threadsperblock = [x, y]
        #blockspergrid_x=1
        #blockspergrid_y=1
        #temp=np.zeros(4)
        if(x>TPB):
            threadsperblock[0]=TPB
            
        if(y>TPB):
            threadsperblock[1]=TPB
        threadsperblock=(threadsperblock[0],threadsperblock[1])
        blockspergrid_x = int(math.ceil(x / threadsperblock[0]))    
        blockspergrid_y = int(math.ceil(y / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #C_global_mem = cuda.device_array((w.shape[0], a.shape[1]))
        #generate random number object to use in kernel function.
        rng_states = create_xoroshiro128p_states(x*y, seed=random.randint(0, 10))
        #print(nn.biases[u],"initial")
        #print(blockspergrid,threadsperblock,rng_states.shape)
        biase_mutation[blockspergrid, threadsperblock](rng_states,nn.biases[u],self.mutation_rate)    
        #print(nn.biases[u],"final",temp)
    #weight update for layer 
    for u in range(nn.num_layers-1):
        #layer, point = get_random_point(self,'weight')
        #if random.uniform(0,1) < self.mutation_rate:
        #    nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
        TPB=4
        x=nn.weights[u].shape[0]
        y=nn.weights[u].shape[1]
        threadsperblock = [x, y]
        #blockspergrid_x=1
        #blockspergrid_y=1
        if(x>TPB):
            threadsperblock[0]=TPB
            
        if(y>TPB):
            threadsperblock[1]=TPB
        threadsperblock=(threadsperblock[0],threadsperblock[1])
        blockspergrid_x = int(math.ceil(x / threadsperblock[0]))    
        blockspergrid_y = int(math.ceil(y / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #generate random number object to use in kernel function.
        rng_states = create_xoroshiro128p_states(x*y, seed=random.randint(0, 10))

        weight_mutation[blockspergrid, threadsperblock](rng_states,nn.weights[u],self.mutation_rate)
        #print(nn.weights[u],"final")
    return nn

#this function is to call for the crossover and mutation based on the fitness value of all the chromosomes in the population.
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


#object to create chromosomes for population.
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

    #number of population that should be maintained
    N_POPS = 15
    #number of nodes in each layer
    NET_SIZE = [24,100,50,2]
    #mutaion rate 
    MUTATION_RATE = 0.4
    #crossover rate
    CROSSOVER_RATE = 0.6
    #retaintion rate that has to be maintained for next generation.
    RETAIN_RATE = 0.4

    #object to create the network(chromosomes) in  population
    nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, X, y)

    start_time = time.time()
   
    i=0
    temp_accuracy=0
    #number of epoch.
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
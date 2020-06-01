from __future__ import division
from numba import cuda
import random
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)
def relu(z):
        return np.maximum(z, 0)
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
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
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
    for _ in range(self.nets[0].bias_nitem):
        layer, point = get_random_point(self,'bias')
        if random.uniform(0,1) < self.crossover_rate:
            nn.biases[layer][point] = mother.biases[layer][point]

    for _ in range(self.nets[0].weight_nitem):
        layer, point = get_random_point(self,'weight')
        if random.uniform(0,1) < self.crossover_rate:
            nn.weights[layer][point] = mother.weights[layer][point]
    
    return nn
    
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
   
    epoch=100
    temp_accuracy=0
    i=0
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
    print("sequential Execution time in seconds ",(time.time() - start_time))
if __name__ == "__main__":
    main()
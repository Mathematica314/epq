import random
import math
import pickle

SIGMOID_FACTOR = 1

def sigmoid(x):
    try:
        return 1/(1+math.exp(-x*SIGMOID_FACTOR))
    except:
        if x<0:
            return 0
        return 1

def inv_sigmoid(x):
    if x==1:
        return 2**64 - 1
    return math.log(x) - math.log(1-x)

def sigmoid_prime(x):
    try:
        return (SIGMOID_FACTOR*math.exp(-x*SIGMOID_FACTOR))/((1+math.exp(-x*SIGMOID_FACTOR))**2)
    except:
        return 0

def relu_prime(x):
    return int(x>=0)

def softmax(vals):
    vals = [math.exp(v) for v in vals]
    return [v/sum(vals) for v in vals]

class Net:
    def __init__(self,structure,classes):
        self.structure = structure
        self.classes = classes
        self.biases = []
        self.weights = []
        for i,l in enumerate(self.structure):
            if i > 0:
                self.biases.append([1-2*random.random() for _ in range(l)])
            if i < len(self.structure) - 1:
                self.weights.append([[1-2*random.random() for _ in range(l)] for _ in range(self.structure[i+1])])
    def layer(self,prev,index,activation):
        new = [sum([x*node for x,node in zip(w,prev)]) for w in self.weights[index]]
        new = [activation(x+b) for x,b in zip(new,self.biases[index])]
        return new
    def net(self,values):
        for i,_ in enumerate(self.structure):
            if i < len(self.structure) - 1:
                values = self.layer(values, i, sigmoid)
        return values
    def fullnet(self,values):
        acts = []
        for i,_ in enumerate(self.structure):
            acts.append(values.copy())
            if i < len(self.structure) - 1:
                values = self.layer(values, i, sigmoid)
        return acts
    def backprop(self,example):
        values = self.fullnet(example[1])
        desired = [1 if example[0]==i else 0 for i in self.classes]
        weight_changes = []
        bias_changes = []
        del_value = [2*(v-d) for v,d in zip(values[-1],desired)]
        for l_index in range(len(self.structure[::-1])-1):
            noact = self.layer(values[-(l_index+2)],len(self.weights)-l_index-1,lambda x:x)
            del_bias = [sigmoid_prime(noact[i])*del_value[i] for i,x in enumerate(values[-(l_index+1)])]
            del_weight = []
            del_prev = []
            weights = self.weights[-(l_index+1)]
            for n_index,neuron in enumerate(weights):
                del_weight.append([del_bias[n_index]*values[-(l_index+2)][i] for i in range(len(neuron))])
            for p_wts in zip(*weights[::-1]):
                del_prev.append(sum([del_bias[i] * x for i,x in enumerate(p_wts)]))
            weight_changes.append(del_weight.copy())
            bias_changes.append(del_bias.copy())
            del_value = del_prev.copy()
        return weight_changes[::-1],bias_changes[::-1]
    def epoch(self,examples,lr):
        weight_changes = [[[0 for i in j] for j in k] for k in self.weights]
        bias_changes = [[0 for i in j] for j in self.biases]
        for e in examples:
            w,b = self.backprop(e)
            for i_x,x in enumerate(w):
                for i_y,y in enumerate(x):
                    for i_z,z in enumerate(y):
                        weight_changes[i_x][i_y][i_z] += z
            for i_x,x in enumerate(b):
                for i_y,y in enumerate(x):
                    bias_changes[i_x][i_y] += y
        self.weights = [[[a + (i*lr)/len(examples) for a,i in zip(b,j)] for b,j in zip(c,k)] for c,k in zip(self.weights,weight_changes)]
        self.biases = [[a + (i*lr)/len(examples) for a,i in zip(b,j)] for b,j in zip(self.biases,bias_changes)]
    def test(self,examples):
        loss = 0
        acc = 0
        for e in examples:
            l,a = self.testpoint(e)
            loss += l
            acc += int(a)
        return loss/len(examples),acc/len(examples)
    def testpoint(self,e):
        result = self.net(e[1])
        loss = sum([(1-x)**2 if c == e[0] else x**2 for x,c in zip(result,self.classes)])
        acc = self.classes[result.index(max(result))] == e[0]
        return loss, acc

def test_pytorch():
    from generalised import GeneralNeuralNetwork

    composers = ["bach", "beethoven", "chopin", "haydn", "joplin", "mozart", "scarlatti"]

    n = Net([19,256,256,256,7],composers)
    with open("mlp.pkl","rb") as file:
        ptmodel = pickle.load(file).state_dict()
    p = [ptmodel[x].tolist() for x in list(ptmodel)]

    n.weights = [p[0],p[2],p[4],p[6]]
    n.biases = [p[1],p[3],p[5],p[7]]

    with open("fets.pkl", "rb") as file:
        data = pickle.load(file)

    data = [(composers.index(d[0]),d[1]) for d in data]
    correct = 0
    for row in data:
        result = n.net(row[1])
        print(result)
        success = result.index(max(result))==row[0]
        correct += int(success)
        if not success:
            print(composers[row[0]],composers[result.index(max(result))])

    print(correct/len(data))

with open("fets.pkl","rb") as file:
    data = pickle.load(file)

composers = ["bach", "beethoven", "chopin", "haydn", "joplin", "mozart", "scarlatti"]

data = [(d[0], [x for i, x in enumerate(d[1]) if i not in [1, 4, 6, 13, 15, 16]]) for d in data]

random.shuffle(data)
test = data[:150]
train = data[150:]

n = Net([len(data[0][1]),256,128,len(composers)],composers)

for epc in range(10):
    print(epc)
    print(n.test(test))
    print(n.test(train))
    n.epoch(train,0.000001)
    print(n.fullnet(test[0][1])[3])
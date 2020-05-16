"""
neural net with weights mutation
"""
from random import randint
from copy import deepcopy
from math import tanh


def sum_matrix(matrix):
    deg = 2 if isinstance(matrix[0], list) else 1
    if deg == 1:
        total = 0
        for num in matrix:
            total += num
        return total
    if deg == 2:
        totals = []
        for row in matrix:
            total = 0
            for num in row:
                total += num
            totals.append(total)
        return totals


def dot_product(matrix1, matrix2):
    assert isinstance(matrix1[0], list)
    assert isinstance(matrix2[0], (int, float))
    totals = []
    for row_num in range(matrix1.__len__()):
        total = 0
        for col_num in range(matrix1[row_num].__len__()):
            total += matrix2[col_num] * matrix1[row_num][col_num]
        totals.append(total)
    return totals


class Layer:
    def __init__(self, nodes_in, nodes_out, activation_function=tanh):
        self.weights_per_node = nodes_in
        self.nodes_num = nodes_out
        self.activation_func = activation_function
        self.activations = []
        self.biases = []
        for _ in range(self.nodes_num):
            self.activations.append(0)
            self.biases.append(0)
        self.weights = []
        for node_num in range(self.nodes_num):
            self.weights.append([])
            for weight_num in range(self.weights_per_node):
                self.weights[node_num].append(0)

    def print_info(self):
        print("weights_per_node:\t", self.weights_per_node)
        print("nodes_num:\t", self.nodes_num)
        print("activation_func:\t", self.activation_func)
        print("activations:\t", self.activations)
        print("weights:\t", self.weights)
        print("biases:\t", self.biases)

    def randomize_weights(self):
        for node_num in range(self.weights.__len__()):
            for weight_num in range(self.weights[node_num].__len__()):
                self.weights[node_num][weight_num] = randint(-1000, 1000) / 1000

    def feed_forward(self, input_activations):
        self.activations = dot_product(self.weights, input_activations)
        self.activations = list(map(self.activation_func, self.activations))

    def mutate_layer(self, rate):
        assert 0 <= rate <= 1
        for node_num in range(self.weights.__len__()):
            for weight_num in range(self.weights[node_num].__len__()):
                random_num = randint(0, 10000000000000000)/10000000000000000
                if random_num < rate:
                    self.weights[node_num][weight_num] += (random_num if randint(1, 10) > 5 else -random_num)
        for bias_num in range(self.biases.__len__()):
            random_num = randint(0, 10000000000000000) / 10000000000000000
            if random_num < rate:
                self.biases[bias_num] += (random_num if randint(1, 10) > 5 else -random_num)/2


class Network:
    def __init__(self, layers=None):
        self.hidden_layers_num = layers.__len__() - 1
        self.layers = []
        self.fitness = None
        self.loss = None
        self.outputs = None
        for layer_index in range(layers.__len__() - 1):
            self.layers.append(Layer(layers[layer_index], layers[layer_index + 1]))

    def print_info(self):
        print("\n")
        print("fitness:\t", self.fitness)
        print("loss:\t", self.loss)
        for layer_index in range(self.layers.__len__()):
            print("layer num:\t", layer_index)
            self.layers[layer_index].print_info()
            print("")

    def randomize_weights(self):
        for layer in self.layers:
            layer.randomize_weights()

    def randomize_biases(self):
        for layer in self.layers:
            for bias_index in range(layer.biases.__len__()):
                layer.biases[bias_index] = randint(-1000, 1000) / 1000

    def feed(self, input_activations):
        self.layers[0].feed_forward(input_activations)
        for layer_index in range(1, self.layers.__len__()):
            self.layers[layer_index].feed_forward(self.layers[layer_index - 1].activations)
        self.outputs = self.layers[self.layers.__len__() - 1].activations
        return self.outputs

    def mutate(self, rate):
        for layer in self.layers:
            layer.mutate_layer(rate)


def calculate_fitness_for_xor(net: Network, inputs, actions):
    loss = 0
    for index in range(inputs.__len__()):
        net.feed(inputs[index])
        loss += abs(net.outputs[0] - actions[index])
    net.loss = loss


xor_in = [
    [1, 1],
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]
xor_out = [0, 0, 1, 1, 0]
networks = []
best_fitness_index = 0
for i in range(5000):
    networks.append(Network(layers=[2, 3, 1]))
    networks[i].randomize_weights()
    calculate_fitness_for_xor(networks[i], xor_in, xor_out)
    if networks[best_fitness_index].loss > networks[i].loss:
        best_fitness_index = i
networks[best_fitness_index].print_info()

"""
# keep parents
def generation(parent):
    global networks, best_fitness_index
    best_fitness_index = 0
    networks.clear()
    networks.append(deepcopy(parent))
    for _ in range(1, 100):
        networks.append(deepcopy(parent))
        networks[_].mutate(0.05)
        calculate_fitness_for_xor(networks[_], xor_in, xor_out)
        if networks[best_fitness_index].loss > networks[_].loss:
            best_fitness_index = _
    networks[best_fitness_index].print_info()
"""


def generation(parent):
    global networks, best_fitness_index
    best_fitness_index = 0
    networks.clear()
    for _ in range(0, 5000):
        networks.append(deepcopy(parent))
        networks[_].mutate(0.05)
        calculate_fitness_for_xor(networks[_], xor_in, xor_out)
        if networks[best_fitness_index].loss > networks[_].loss:
            best_fitness_index = _
    networks[best_fitness_index].print_info()


for generation_num in range(1000):
    generation(networks[best_fitness_index])

n = networks[best_fitness_index]
print(xor_in)
print(xor_out)
for inp in xor_in:
    print(n.feed(inp))

"""
Note:
    Matrices are like:
        ( [1, 2] )
        | [....] |
        |        |
        (        )
    matrix = [[1, 2], [...]]
"""

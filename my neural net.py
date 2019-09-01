"""
created by Moe Ramzi Assal at 8/20/.
python 3.3+
still under development
contact:
+96171804948
mohammad.elassal04@gmail.com
"""

from random import randint
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def point_loss(self, network):
        network.update_inputs([self.x])
        network.compute_whole_network()
        network_y = network.layer_activations(network.output_layer_num)[0]
        loss = (float(self.y) - float(network_y)) ** 2
        return loss


class Node:
    def __init__(self, node_num, layers):
        self.node_num = node_num
        self.activation = 0
        self.weights = []
        self.bias = 0
        self.layer_num = None
        total_nodes = -1    # not 0 to make the counter start from 0. 2 lines after.
        for layer_num in range(len(layers)):    # to know the node's layer from its number
            for layer_nodes in range(layers[layer_num]):
                total_nodes = total_nodes + 1
                if self.node_num == total_nodes:
                    self.layer_num = layer_num
                    break
        if self.layer_num != 0:     # to avoid locating weights for input nodes
            self.is_input = False
            for nodes in range(layers[self.layer_num - 1]):
                self.weights.append(randint(-1500, 1500) / 1000)      # pick best random
        else:
            self.weights = None
            self.is_input = True


class Network:
    def __init__(self, layers):
        self.layers = layers    # layers must be passed as a tuple or list or array
        self.layers_num = len(layers)
        self.nodes = []
        self.total_nodes = -1
        for layer_nodes in self.layers:
            for node_num in range(layer_nodes):
                self.total_nodes = self.total_nodes + 1
                self.nodes.append(Node(self.total_nodes, self.layers))
        self.total_nodes = self.total_nodes + 1
        self.output_layer_num = self.layers_num - 1     # initialization is completed here

        self.points = []
        self.delta = 0.00001
        self.learning_rate = 0.01
        self.no_change = 0
        self.weights_num = 0
        for node in self.nodes:
            if not node.is_input:
                self.weights_num += len(node.weights)

    def compute_activation(self, node_num):
        # activation is automatically updated for "node_num" node
        if not self.nodes[node_num].is_input:
            weights = self.nodes[node_num].weights
            bias = self.nodes[node_num].bias
            layer_num = self.nodes[node_num].layer_num
            layer_node_num = self.layers[layer_num - 1]
            activations = self.layer_activations(layer_num - 1)
            total = 0 + bias
            for calculation_num in range(layer_node_num):
                total = total + weights[calculation_num] * activations[calculation_num]

            if self.nodes[node_num].activation < 0:
                self.nodes[node_num].activation = 0.01 * total  # activation function
            else:
                self.nodes[node_num].activation = total
            return True
        else:
            return False

    def update_inputs(self, input_nodes_activation):
        for input_node_num in range(self.layers[0]):
            self.nodes[input_node_num].activation = input_nodes_activation[input_node_num]
        return True

    def compute_whole_network(self):
        for node in self.nodes:
            self.compute_activation(node.node_num)

    def layer_activations(self, layer_num):
        activations = []
        for node in self.nodes:
            if node.layer_num == layer_num:
                activations.append(node.activation)
        return activations

    def change_weight_by(self, delta, node_num, weight_num):
        self.nodes[node_num].weights[weight_num] = self.nodes[node_num].weights[weight_num] + delta
    # this is a basic neural network configuration, modify the below as you need to make your unique network

    def create_points(self, x_values, y_values):
        for i in range(len(x_values)):
            self.points.append(Point(x_values[i], y_values[i]))

    def loss(self):
        general_loss = 0
        for point in self.points:
            general_loss = general_loss + point.point_loss(self)
        return general_loss

    def train_weight(self, node_num, weight_num):
        try:
            self.compute_whole_network()
            loss_initial = self.loss()
            self.change_weight_by(self.delta, node_num, weight_num)
            self.compute_whole_network()
            loss_final = self.loss()
            self.change_weight_by(-self.delta, node_num, weight_num)
            update = self.delta / (loss_initial - loss_final)
            update = update * self.learning_rate
            self.change_weight_by(update, node_num, weight_num)
            if loss_final < self.loss() - 1:
                self.change_weight_by(-update, node_num, weight_num)
                self.no_change += 1
                print(str(update) + "\t" + str(loss_initial) + "\t" + str(loss_final) + "\t" + str(self.loss())
                      + "\tno change")
                return False
            print(str(update) + "\t" + str(loss_initial) + "\t" + str(loss_final) + "\t" + str(self.loss()))
            return True
        except ZeroDivisionError:
            print("0 d error")

    def train_node(self, node_num):
        for weight_index in range(len(self.nodes[node_num].weights)):
            self.train_weight(node_num, weight_index)
        return True

    def train_network(self, iterations):
        for iteration in range(iterations):
            for node_num in range(self.layers[0], len(self.nodes)):
                self.train_node(node_num)
            if self.no_change == self.weights_num:
                return False
            else:
                self.no_change = 0

        return True

    def return_weights(self):
        weights = []
        for node in self.nodes:
            weights.append(node.weights)
        return weights


def pyplot_function(points_x, points_y, network):
    min_x = int(min(points_x) - 5)
    max_x = int(max(points_x) - 5) * 2
    print(min_x)
    network_y = []
    network_x = []
    for x_range in range(min_x * 10, max_x * 10):
        atomic_x = x_range / 10
        network_x.append(atomic_x)
        network.update_inputs([atomic_x])
        network.compute_whole_network()
        value = network.layer_activations(network.output_layer_num)[0]
        network_y.append(value)
    plt.plot(points_x, points_y, marker='.', linestyle='', color='r')
    plt.plot(network_x, network_y, marker='', linestyle='-', color='b')
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("graph")
    plt.grid(True)
    plt.show()
    return True


if __name__ == '__main__':
    # this is a simple model
    my_net = Network([1, 5, 5, 4, 4, 4, 1])
    x_ = [4, 8, 2, 5, 1, 11, 3]
    y_ = [7, 9, 4, 3, 1, 5, 6]
    my_net.create_points(x_, y_)
    pyplot_function(x_, y_, my_net)
    print(my_net.train_network(10))
    pyplot_function(x_, y_, my_net)

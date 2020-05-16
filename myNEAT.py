"""
M-NEAT algorithm

TODO:
    - Add Innovation Numbers as a global variable
    - Test
        -copy genotype appropriately in Genetic and Creature and Crossover
    - Documentation

nice genotype for xor:
    [[['0 3', -17.597303078000003, True], ['1 3', -16.427530955, True], ['0 4', 4.763314708, True],
    ['1 4', -1.9906530880000002, True], ['3 2', -6.833571253000001, True], ['4 2', -2.670385535, True],
    ['1 2', -4.183899875201467, True]], [0, 1, 2, 3, 4], [0, 1], [2]]
"""
from math import tanh, floor
from random import randint, random, uniform
# indexes in the genotype[0] for 1 connection
CONN_NODES = 0
WEIGHT = 1
ACTIVE = 2

# indexes in genotype
GENETIC_INFO = 0
NODES = 1
INPUTS = 2
OUTPUTS = 3
geno = [
    [
        [
            "0 3",
            -2.60,
            True
        ],
        [
            "1 3",
            -2.71,
            True
        ],
        [
            "0 4",
            2.29,
            True
        ],
        [
            "1 4",
            -1.11,
            True
        ],
        [
            "3 2",
            -1.84,
            True
        ],
        [
            "4 2",
            -1.76,
            True
        ]
    ],
    [
        0,
        1,
        2,
        3,
        4
    ],
    [
        0,
        1
    ],
    [
        2
    ]
]


class Node:
    def __init__(self, node_num, is_input=False):
        self.nodes_from = []
        self.weights = []
        self.active_nodes = []
        self.calculated = False
        self.is_input = is_input
        self.node_num = node_num
        self.activation = 0
        self.activation_function = tanh     # user defined

    def need_activation(self):
        if self.calculated or self.is_input:
            return self.activation
        else:
            self.calculate_activation()
            return self.activation

    def add_connection(self, node_from, weight, active):
        self.nodes_from.append(node_from)
        self.weights.append(weight)
        self.active_nodes.append(active)

    def calculate_activation(self):
        total = 0
        for node_index in range(self.nodes_from.__len__()):
            if self.active_nodes[node_index]:
                total += self.nodes_from[node_index].need_activation() * self.weights[node_index]
        self.activation = self.activation_function(total)
        self.calculated = True


class Network:
    def __init__(self, genotype):
        self.nodes = []
        self.inputs = genotype[INPUTS]
        self.outputs = genotype[OUTPUTS]
        for node_num in genotype[1]:
            if node_num in self.inputs:
                self.nodes.append(Node(node_num=node_num, is_input=True))
            else:
                self.nodes.append(Node(node_num=node_num))
        for conn in genotype[0]:
            n1, n2 = conn[CONN_NODES].split(" ")
            n1, n2 = int(n1), int(n2)
            self.nodes[n2].add_connection(self.nodes[n1], conn[WEIGHT], conn[ACTIVE])

    def feed(self, input_activations):
        output_activations = []
        for input_node_index in range(self.inputs.__len__()):
            self.nodes[input_node_index].activation = input_activations[input_node_index]
        for output_node in self.outputs:
            output_activations.append(self.nodes[output_node].need_activation())

        for node in self.nodes:
            node.calculated = False
        return output_activations


class Genetic:
    def __init__(self, genotype=None, genetic=None, ):
        if isinstance(genetic, Genetic):
            self.genotype = genetic.genotype.copy()
        elif isinstance(genotype, list):
            self.genotype = genotype
        else:
            raise TypeError

    def mutate_weights(self, rate, _max_=0.5):  # rate is between 0 and 1
        for conn in self.genotype[GENETIC_INFO]:
            if random() <= rate:
                conn[WEIGHT] += uniform(-_max_, _max_)

    def mutate_state(self, rate):
        for conn in self.genotype[GENETIC_INFO]:
            if random() <= rate:
                conn[ACTIVE] = not conn[ACTIVE]

    def connection_available(self, node_conn):
        for conn in self.genotype[GENETIC_INFO]:
            if conn[CONN_NODES] == node_conn:
                return False
        return True

    def mutate_connections(self, rate):
        for node in self.genotype[NODES]:
            if random() <= rate:
                random_node = randint(0, self.genotype[NODES].__len__() - 1)
                if self.connection_available(str(node) + " " + str(random_node)) \
                        and random_node != node \
                        and self.connection_available(str(random_node) + " " + str(node))\
                        and random_node not in self.genotype[INPUTS] \
                        and node not in self.genotype[OUTPUTS]:
                    self.genotype[GENETIC_INFO].append([str(node) + " " + str(random_node),
                                                        random() - 0.5, True])   # user defined
                    try:
                        n = Network(self.genotype)
                        ins = [0] * self.genotype[INPUTS].__len__()
                        n.feed(ins)
                    except RecursionError:
                        self.genotype[GENETIC_INFO].pop()
                        continue

    def mutate_nodes(self, rate):
        for conn_address, conn in enumerate(self.genotype[GENETIC_INFO]):
            if random() <= rate:
                n1, n2 = conn[CONN_NODES].split(" ")
                n3 = self.genotype[NODES].__len__()
                self.genotype[NODES].append(n3)
                self.genotype[GENETIC_INFO].append([str(n1) + " " + str(n3), 1., True])    # user defined
                self.genotype[GENETIC_INFO].append([str(n3) + " " + str(n2), conn[WEIGHT], True])
                self.genotype[GENETIC_INFO].pop(conn_address)

    @staticmethod
    def cross_over(parentx, parenty):
        while True:
            offspring_genotype = [[],
                                  parentx.genetic.genotype[INPUTS].copy() + parentx.genetic.genotype[OUTPUTS].copy(),
                                  parentx.genetic.genotype[INPUTS].copy(),
                                  parentx.genetic.genotype[OUTPUTS].copy()
                                  ]
            for block in parentx.genetic.genotype[GENETIC_INFO] + parenty.genetic.genotype[GENETIC_INFO]:
                if random() > 0.5:
                    offspring_genotype[GENETIC_INFO].append(block.copy())
            for block in offspring_genotype[GENETIC_INFO]:
                n0, n1 = block[CONN_NODES].split(" ")
                n0, n1 = int(n0), int(n1)
                if n0 not in offspring_genotype[NODES]:
                    offspring_genotype[NODES].append(n0)
                if n1 not in offspring_genotype[NODES]:
                    offspring_genotype[NODES].append(n1)
            # genotype correction
            offspring_genotype[NODES] = sorted(offspring_genotype[NODES])
            for i, node_num in enumerate(offspring_genotype[NODES]):
                if node_num != i:
                    offspring_genotype[NODES][i] = i
                    for block in offspring_genotype[GENETIC_INFO]:
                        n0, n1 = block[CONN_NODES].split(" ")
                        n0, n1 = int(n0), int(n1)
                        if node_num == n0:
                            block[CONN_NODES] = str(i) + " " + str(n1)
                        if node_num == n1:
                            block[CONN_NODES] = str(n0) + " " + str(i)
            pop_them = []
            for index, block in enumerate(offspring_genotype[GENETIC_INFO]):
                for i in range(index + 1, offspring_genotype[GENETIC_INFO].__len__()):
                    if block[CONN_NODES] == offspring_genotype[GENETIC_INFO][i][CONN_NODES]:
                        pop_them.append(i)
            pop_them = sorted(list(set(pop_them)), reverse=True)
            for index in pop_them:
                offspring_genotype[GENETIC_INFO].pop(index)
            try:
                n = Network(offspring_genotype)
                ins = [1] * offspring_genotype[INPUTS].__len__()
                n.feed(ins)
            except RecursionError:
                continue
            return Creature(genotype=offspring_genotype)

    @staticmethod
    def first_genetic(inputs_num, outputs_num):
        genotype = [
            [],
            [],
            [],
            []
        ]
        for node in range(inputs_num + outputs_num):
            genotype[NODES].append(node)
        for input_node in range(inputs_num):
            genotype[INPUTS].append(input_node)
        for output_node in range(inputs_num, inputs_num + outputs_num):
            genotype[OUTPUTS].append(output_node)
        for input_node in range(inputs_num):
            for output_node in range(inputs_num, inputs_num + outputs_num):
                genotype[GENETIC_INFO].append([str(input_node) + " " + str(output_node), random(), random() < 0.5])
        return Genetic(genotype=genotype)

    def mutate(self):   # user defined
        self.mutate_weights(0.4, 0.5)
        self.mutate_state(0.01)
        self.mutate_connections(0.1)
        self.mutate_nodes(0.05)
        try:
            n = Network(self.genotype)
            n.feed([0] * self.genotype[INPUTS].__len__())
        except RecursionError:
            print(self.genotype)
            raise RecursionError


xors = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]


class Creature:
    def __init__(self, genotype=None, creature=None, genetic=None, ):
        if isinstance(creature, Creature):
            self.genetic = Genetic(genetic=creature.genetic)
        elif isinstance(genotype, list):
            self.genetic = Genetic(genotype=genotype)
        elif isinstance(genetic, Genetic):
            self.genetic = Genetic(genetic=genetic)
        else:
            raise TypeError

        self.net = Network(self.genetic.genotype)
        self.score = None
        self.fitness = None
        self.genetic_drift = None
        self.pool = None

    def decision(self, inputs):  # TODO: implement softmax
        outs = self.net.feed(inputs)
        max_ = outs[0]
        best = 0
        for index, out in enumerate(outs):
            if max_ < out:
                max_ = out
                best = index
        return [best, outs[best]]

    def mutate(self):
        self.genetic.mutate()
        self.net = Network(self.genetic.genotype)


class Pool:
    def __init__(self, num):
        self.parents = []
        self.size = 0
        self.creatures = []
        self.a_fitness = None
        self.generation = 0
        self.num = num
        self.best_offsprings = []

    def compute_fitnesses(self, fitness_func):
        for creature in self.creatures:
            try:
                creature.fitness = fitness_func(creature)
            except RecursionError:
                print(creature.genetic.genotype)
                raise RecursionError

    def average_fitness(self):
        _sum_ = 0
        for creature in self.creatures:
            _sum_ += creature.fitness
        self.a_fitness = _sum_ / self.size
        return self.a_fitness

    def best_children(self):
        max_fit = self.creatures[0].fitness
        best = 0
        for i, creature in enumerate(self.creatures):
            if creature.fitness > max_fit:
                max_fit = creature.fitness
                best = i
        child_0 = self.creatures[best]
        try:
            max_fit = self.creatures[0].fitness
            best = 0
            for i, creature in enumerate(self.creatures):
                if creature.fitness > max_fit and id(creature) != id(child_0):
                    max_fit = creature.fitness
                    best = i
            child_1 = self.creatures[best]
            self.best_offsprings = [child_0, child_1]
        except IndexError:
            self.best_offsprings = [child_0, child_0]
        return self.best_offsprings


def fit():
    return 0


class Neat:
    def __init__(self, config):
        self.total_size = config['Population']
        self.f_func = config['fitness function']
        self.conn_drift = config['conn drift']
        self.node_drift = config['node drift']
        self.diversity = config['diversity']
        self.pop_threshold = config['population threshold']
        self.inputs_num = config['Inputs']
        self.outputs_num = config['Outputs']
        self.generation_reached = 0
        self.pools = []
        self.past_generation = None
        self.base_drift = config['node drift'] * (config['Inputs'] + config['Outputs'])

    def initialize_model(self):
        self.generation_reached += 1
        first_pool = Pool(0)
        first_pool.size = self.total_size
        self.pools.append(first_pool)
        gen0 = Generation(self.total_size, self.pools, self.pop_threshold, self.conn_drift, self.node_drift,
                          self.base_drift, self.diversity)
        for i in range(self.total_size):
            gen0.creatures.append(Creature(genetic=Genetic.first_genetic(self.inputs_num, self.outputs_num)))
        gen0.speciate_creatures()
        gen0.compute_pools_fitnesses(self.f_func)
        self.past_generation = gen0

    def next_generation(self):
        self.generation_reached += 1
        next_gen = Generation(self.total_size, self.pools, self.pop_threshold, self.conn_drift, self.node_drift,
                              self.base_drift, self.diversity)
        next_gen.extract_old_info()
        next_gen.create_creatures()
        next_gen.mutate_creatures()
        next_gen.speciate_creatures()
        next_gen.compute_pools_fitnesses(self.f_func)
        self.past_generation = next_gen
        return next_gen

    def best_creature(self):
        best = self.pools[0].best_children()[0]
        for pool in self.pools:
            if pool.best_children()[0].fitness > best.fitness:
                best = pool.best_offsprings[0]
        return best


class Generation:
    def __init__(self, pop_size, pools, pop_threshold, conn_drift, node_drift, base_drift, diversity):
        self.pop_size = pop_size
        self.pools = pools
        self.total_creatures = []
        self.all_parents = []
        self.pop_threshold = pop_threshold
        self.creatures = []
        self.conn_drift = conn_drift
        self.node_drift = node_drift
        self.base_drift = base_drift
        self.diversity = diversity

    def compute_pools_fitnesses(self, f):
        for pool in self.pools:
            if isinstance(pool, Pool):
                pool.compute_fitnesses(f)

    def extract_old_info(self):
        _sum_ = 0
        for pool in self.pools:
            if isinstance(pool, Pool):
                self.all_parents.append(pool.best_children())
                _sum_ += pool.average_fitness()
        for pool in self.pools:
            if isinstance(pool, Pool):
                pool.size = int(pool.a_fitness * self.pop_size / _sum_)
                if pool.size < self.pop_threshold and pool.generation > 15:   # revise
                    self.pools[pool.num] = pool.best_offsprings[0]

    def create_creatures(self):
        self.creatures.clear()
        for parents in self.all_parents:
            pool = self.pools[parents[0].pool]
            # collect garbage
            for i in range(pool.size):
                self.creatures.append(Genetic.cross_over(parents[0], parents[1]))

    def speciate_creatures(self):
        self.compute_genetic_drifts()
        for pool in self.pools:
            pool.creatures.clear()
        for creature in self.creatures:
            creature.pool = int(floor(creature.genetic_drift / self.diversity))
            try:
                self.pools[creature.pool].creatures.append(creature)
            except IndexError:
                if self.pools[-1].num == creature.pool - 1:
                    self.pools.append(Pool(creature.pool))
                    self.pools[creature.pool].creatures.append(creature)
            except AttributeError:
                continue
        for pool in self.pools:
            pool.size = pool.creatures.__len__()

    def mutate_creatures(self):
        for creature in self.creatures:
            creature.mutate()

    def compute_genetic_drifts(self):
        for creature in self.creatures:
            creature.genetic_drift = self.conn_drift * creature.genetic.genotype[GENETIC_INFO].__len__() + \
                                     self.node_drift * creature.genetic.genotype[NODES].__len__() - self.base_drift

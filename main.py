import pickle
import time
from copy import deepcopy
from random import random, randrange

import seaborn as sns
from matplotlib import pyplot as plt

WIDTH, HEIGHT = 8, 8
FOOD_ON_FIELD = 12
START_BALANCE = 10
EPOCHS = 100
ANTS_PER_ANT = 50_000
EVOLUTION_COEFFICIENT = 0.3
MUTATION_PROBABILITY = 0.5
CROSSOVER_PROBABILITY = 0.3
VIEW_SIZE = 5*5
LIFE_PER_ANT = 10

LOAD_ANT = 0


class Ant:
    def __init__(self):
        global WIDTH, HEIGHT, VIEW_SIZE

        self.input_layer = VIEW_SIZE
        self.output_layer = 4

        self.layers = [[random()] * (self.input_layer * self.output_layer)]

    @staticmethod
    def process(field):
        global WIDTH, HEIGHT, VIEW_SIZE

        result = [-1 for _ in range(VIEW_SIZE)]

        x, y = 0, 0
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if field[i][j] == 0.5:
                    x = j
                    y = i
        mid = (VIEW_SIZE - 1) // 2
        sqr = int(VIEW_SIZE ** 0.5)
        for k in range(9):
            if 0 <= x + k % sqr < WIDTH:
                if 0 <= y + k // sqr < HEIGHT:
                    result[k - (k > mid)] = field[y + k // sqr][x + k % sqr]

        return result

    def move(self, field):
        network = [self.process(field), [0, 0, 0, 0]]

        for p in range(len(network) - 1):
            for f in range(len(network[p + 1])):
                for s in range(len(network[p])):
                    network[p + 1][f] += network[p][s] * self.layers[p][f * len(network[p + 1]) + s]

        result = 0
        for i in range(len(network[-1])):
            if network[-1][i] > network[-1][result]:
                result = i

        return result

    def evolve(self, other=None):
        global EVOLUTION_COEFFICIENT
        new_ant = Ant()
        new_ant.layers = deepcopy(self.layers)

        if other is not None:
            for layer in range(len(new_ant.layers)):
                for i in range(len(new_ant.layers[layer])):
                    if random() < CROSSOVER_PROBABILITY:
                        new_ant.layers[layer][i] = other.layers[layer][i]

        for layer in range(len(new_ant.layers)):
            for i in range(len(new_ant.layers[layer])):
                if random() < MUTATION_PROBABILITY:
                    new_ant.layers[layer][i] += EVOLUTION_COEFFICIENT * (2 * random() - 1)

        return new_ant


def generate_field():
    """
    0 - Nothing
    1 - food
    0.5 - ant
    """
    global WIDTH, HEIGHT, FOOD_ON_FIELD

    field = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]

    x, y = randrange(0, WIDTH), randrange(0, HEIGHT)
    field[y][x] = 0.5

    food_count = 0
    while food_count < FOOD_ON_FIELD:
        x, y = randrange(0, WIDTH), randrange(0, HEIGHT)
        if field[y][x] != 0:
            continue

        field[y][x] = 1
        food_count += 1

    return field


def print_field(field):
    icons = {0: ".", 0.5: "x", 1: "@", }
    for i in range(len(field)):
        string = []
        for j in range(len(field[i])):
            string.append(icons[field[i][j]])
        print(" ".join(string))


def life(ant, field, verbose=False):
    global WIDTH, HEIGHT, START_BALANCE

    directions = {0: "->", 1: "<-", 2: "v", 3: "^", }
    score = -START_BALANCE
    balance = START_BALANCE
    new_cells = set()
    x, y = -1, -1
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if field[i][j] == 0.5:
                x, y = j, i

    if verbose:
        print(f"START")
        print_field(field)
        print()

    while balance > 0:
        step = ant.move(field)

        field[y][x] = 0

        old_x = x
        old_y = y

        if step == 0:
            x += 1
        elif step == 1:
            x -= 1
        elif step == 2:
            y += 1
        else:
            y -= 1

        x, y = min(WIDTH - 1, max(0, x)), min(HEIGHT - 1, max(0, y))

        score += 1
        if x == old_x and y == old_y:
            balance -= 2

        if field[y][x] == 1:
            balance += 3

        balance -= 1

        field[y][x] = 0.5

        if f"{x} {y}" not in new_cells:
            new_cells.add(f"{x} {y}")
            balance += 0.5
        else:
            balance -= 0.3

        if verbose:
            print(f"SCORE {score}\tBALANCE {balance}\t{directions[step]}")
            print_field(field)
            input()

    return score


def clear_models():
    import os, shutil
    folder = 'models'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def evolution():
    global EPOCHS, ANTS_PER_ANT, LOAD_ANT

    history = []

    if LOAD_ANT:
        first_ant = pickle.load(open(f"models/model_{LOAD_ANT}", "rb"))
    else:
        first_ant = Ant()
    second_ant = None

    for epoch in range(EPOCHS):
        start_time = time.time()

        fields = [generate_field() for _ in range(LIFE_PER_ANT)]
        generation = [first_ant.evolve(second_ant) for _ in range(ANTS_PER_ANT)]
        scores = [0 for _ in range(ANTS_PER_ANT)]

        for i in range(ANTS_PER_ANT):
            print("Current run - ", round(100 * i / ANTS_PER_ANT), "%", sep="", end="")

            current_scores = []
            for j in range(LIFE_PER_ANT):
                current_scores.append(life(generation[i], deepcopy(fields[j])))
            scores[i] = sum(current_scores) / len(current_scores)

            print("\b" * (len(str(round(100 * i / ANTS_PER_ANT))) + 15), end='')

        best_ant = 0
        second_best_ant = 1
        for i in range(ANTS_PER_ANT):
            if scores[i] > scores[best_ant]:
                second_best_ant = best_ant
                best_ant = i
            if scores[i] > scores[second_best_ant] and i != best_ant:
                second_best_ant = i

        first_ant = generation[best_ant]
        second_ant = generation[second_best_ant]

        eta = (time.time() - start_time) * (EPOCHS - epoch - 1)
        min_sec = f"{round(eta // 60)}:{round(eta % 60)}"
        history.append(scores[best_ant])
        print(f"{round(100 * (epoch + 1) / EPOCHS, 2)}% finished\tETA: {min_sec}\tbest score: {scores[best_ant]}")

        pickle.dump(first_ant, open(f"models/model_{epoch + 1 + LOAD_ANT}", "wb"))

        plt.clf()
        sns.regplot(x=list(range(len(history))), y=history)
        plt.savefig("plot.png")


if __name__ == "__main__":
    ...
    # clear_models()
    # evolution()
    a: Ant = pickle.load(open("models/final_model", "rb"))
    f = generate_field()
    life(a, f, verbose=True)

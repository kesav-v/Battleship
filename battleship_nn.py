import random
import tensorflow as tf
import numpy as np

"""
Reinforcement learning implementation for Battleship.

Details of the reward function and gradient step adapted from:
http://efavdb.com/battleship/

In essence, the neural network is updated at each step with the probabilities
of firing on each square resulting in a hit; these probabilities are learned from
minimizing a cross entropy function that is inversely related to the reward function,
which is proportional to the number of successful hits.

The network quickly learns (hundreds of steps) how to effectively scan 
the entire board for possible locations of ships, and how to sink
entire ships given the first hit.

Major classes:
- BattleshipPlayer: representation of a human player.
- BattleshipAI: representation of an AI player.
- Battleship: class that handles actually playing the game interactively
or for neural network training purposes.
"""

def is_valid(i0, j0, i1, j1, grid):
    n = len(grid)
    if (not (0 <= i0 < n)) or (not (0 <= j0 < n)):
        return False
    if (not (0 <= i1 <= n)) or (not (0 <= j1 <= n)):
        return False
    for i in range(i0, i1):
        if grid[i][j0]:
            return False
    for j in range(j0, j1):
        if grid[i0][j]:
            return False
    return True

def valid(i, j, n):
    return 0 <= i < n and 0 <= j < n

def flatten(i, j, n):
    return i * n + j

def bin(b):
    return 1 if b else 0

def coords(guess):
    row, col = guess[0], guess[1:]
    j = int(col) - 1
    i = ord(row) - ord('A')
    return i, j

def rand_coords(size):
    i = random.randint(0, size - 1)
    j = random.randint(0, size - 1)
    return i, j

def reward_log(hit_log, bsize, ssize):
    """ Discounted sum of future hits over trajectory"""           
    hit_log_weighted = [(item - 
        float(ssize - sum(hit_log[:index])) / float(bsize - index)) * (
        0.9 ** index) for index, item in enumerate(hit_log)]
    return [(0.9 ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]

class BattleshipPlayer:

    sizes = (3, 2)

    def __init__(self, name, size):
        self.prev_moves = []
        self.results = []
        self.reset(name, size)
    
    def reset(self, name, size):
        self.name = name
        self.grid = []
        self.num_ones = 0
        for i in range(size):
            self.grid.append([])
            for j in range(size):
                self.grid[i].append(0)
        self.size = size
        self.ships = []
        self.place_ships()
        self.checked = dict()
        self.last_hit = ()
        self.last_direction = (0, 0)
        self.other_board = [-1 for _ in range(size * size)]

    def seen(self, i, j):
        hit = False
        if self.grid[i][j] == 1:
            hit = True
            self.num_ones -= 1
        self.grid[i][j] = 2
        return hit

    def guess(self, error_log=None, actions=None):
        return coords(input('Enter a guess -> '))

    def update_checked(self, i, j, res):
        # self.checked[(i, j)] = bin(res)
        # if res:
        #     if len(self.last_hit) > 0:
        #         self.last_direction = (i - self.last_hit[0], j - self.last_hit[1])
        #     self.last_hit = (i, j)
        self.other_board[flatten(i, j, self.size)] = bin(res)

    def print_grid(self):
        for row in self.grid:
            s = ''
            for j in row:
                char = '*' if j == 1 else '-'
                s += char + ' '
            print(s)

    def place_ships(self):
        self.num_ones = sum(self.sizes)
        for s in self.sizes:
            while True:
                i = random.randint(0, self.size - 1)
                j = random.randint(0, self.size - 1)
                i1, j1 = i, j
                orientation = 0 if (random.random() < 0.5) else 1
                if orientation:
                    i1 += s
                else:
                    j1 += s
                if is_valid(i, j, i1, j1, self.grid):
                    for i0 in range(i, i1):
                        self.grid[i0][j] = 1
                    for j0 in range(j, j1):
                        self.grid[i][j0] = 1
                    break

class Battleship:

    def __init__(self, p1, p2):
        self.players = [p1, p2]
        n = p1.size
        for p in self.players:
            p.reset('', n)
        self.size = n

    def play(self, training=False):
        turn = 1
        error_log = False
        next_player = 0
        game_over = False
        num_moves = 0
        results = [[], []]
        actions = [[], []]
        boards = [[], []]
        while not game_over:
            num_moves += 1
            turn = next_player
            next_player = (turn + 1) % len(self.players)
            boards[turn].append([i for i in self.players[turn].other_board])
            i, j = self.players[turn].guess(actions=[i for i in actions[turn]], error_log=error_log)
            if not training:
                print('Player', str(turn + 1) + '\'s turn:', (i, j))
            # guess = input('Player ' + str(turn + 1) + ', make a guess: ')
            hit = self.players[next_player].seen(i, j)
            self.players[turn].update_checked(i, j, hit)
            actions[turn].append(flatten(i, j, self.size))
            results[turn].append(bin(hit))
            if not training:
                if hit:
                  print('hit!')
                else:
                  print('miss!')
            if num_moves > 10000:
              print(boards[turn][-1])
              error_log = True
            game_over = self.players[next_player].num_ones == 0
        
        if not training:
            print('Player', turn + 1, 'wins!')
        # for p in self.players:
            # p.print_grid()
            # print(p.checked)
            # print()
        if type(self.players[turn]) == BattleshipAI:
            self.players[turn].update_nn(boards[turn], actions[turn], results[turn])
        return actions, results, boards, turn

class BattleshipAI(BattleshipPlayer):

    ALPHA = 0.06

    def __init__(self, name, size):
        BattleshipPlayer.__init__(self, name, size)
        self.create_nn(size)

    def create_nn(self, n):
        self.BOARD_SIZE = n * n
        self.SHIP_SIZE = sum(self.sizes)
         
        hidden_units = self.BOARD_SIZE
        output_units = self.BOARD_SIZE
         
        self.input_positions = tf.placeholder(tf.float32, shape=(1, self.BOARD_SIZE))
        self.labels = tf.placeholder(tf.int64)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # Generate hidden layer
        W1 = tf.Variable(tf.truncated_normal([self.BOARD_SIZE, hidden_units],
                     stddev=0.1 / np.sqrt(float(self.BOARD_SIZE))))
        b1 = tf.Variable(tf.zeros([1, hidden_units]))
        h1 = tf.tanh(tf.matmul(self.input_positions, W1) + b1)
        # Second layer -- linear classifier for action logits
        W2 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
                     stddev=0.1 / np.sqrt(float(hidden_units))))
        b2 = tf.Variable(tf.zeros([1, output_units]))
        logits = tf.matmul(h1, W2) + b2 
        self.probabilities = tf.nn.softmax(logits)
         
        init = tf.initialize_all_variables()
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, self.labels, name='xentropy')
        self.train_step = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(cross_entropy)
        # Start TF session
        self.sess = tf.Session()
        self.sess.run(init)

    def guess(self, actions=None, error_log=False):
        probs = self.sess.run([self.probabilities],
            feed_dict={self.input_positions:[[i for i in self.other_board]]})[0][0]
        if error_log:
            print(probs, actions)
        probs = [p * (index not in actions) for index, p in enumerate(probs)]
        probs = [p / sum(probs) for p in probs]
        bomb_index = np.random.choice(self.BOARD_SIZE, p=probs)            
        i, j = bomb_index // self.size, bomb_index % self.size
        return i, j

    def update_nn(self, boards, actions, results):
        rewards = reward_log(results, self.BOARD_SIZE, self.SHIP_SIZE)
        for reward, current_board, action in zip(rewards, boards, actions):
            # Take step along gradient
            self.sess.run([self.train_step], 
                feed_dict={self.input_positions:[current_board],
                self.labels:[action], self.learning_rate:self.ALPHA * reward})


def simulate(num_games, p1, p2):
    counts = dict()
    for i in range(num_games):
        if i and i % 100 == 0:
            print(str(i + 1), 'games played')
            print('Results so far:', counts.get(1, 0), 'wins for Player 1,',
                counts.get(2, 0), 'wins for Player 2')
        game = Battleship(p1, p2)
        actions, results, boards, winner = game.play(training=True)
        counts[winner + 1] = counts.get(winner + 1, 0) + 1
    return counts

BOARD_SIZE = 4
# Change these to BattleshipPlayer to play a 2-player human game
p1 = BattleshipAI('A', BOARD_SIZE)
p2 = BattleshipAI('B', BOARD_SIZE)
print(simulate(2000, p1, p2))
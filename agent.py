import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self) -> None:
        self.no_of_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEM) #if > MEM popleft()
        self.model = None
        self.trainer = None

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, over):
        self.memory.append((state, action, reward, next_state, over)) # popleft if MAX_MEMORY is reached

    def train_long_mem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, overs)
        #for state, action, reward, nexrt_state, done in mini_sample:

    def train_short_mem(self, state, action, reward, next_state, over):
        self.trainer.train_step(state, action, reward, next_state, over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        old_state = agent.get_state(game)
        # get move
        final_move = agent.get_action(old_state)
        # action and get new state
        reward, over, score = game.play_step(final_move)
        new_state = agent.get_state(game)
        # train long memmory
        agent.train_short_mem(old_state, final_move, reward, new_state, over)
        # remember
        agent.remember(old_state, final_move, reward, new_state, over)

        if over:
            # train long memmory
            game.reset()
            agent.no_of_games +=1
            agent.train_long_mem()
            
            if score > best_score:
                best_score = score
                #agen model save and plot
            
            print("Game: ",agent.no_of_games,"Score: ",score,"Best: ",best_score)
                



if __name__ == '__main__':
    train()
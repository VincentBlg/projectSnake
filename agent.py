from os import stat_result
import re
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QNet, QTrainer
from plot import plot    


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet(11,256,3)
        self.trainer = QTrainer(self.model , lr=LR, gamma=self.gamma)

    def get_state(self,game):
        head = game.snake[0]
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)
        
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        # state = []                                        State for the model w/ 85 values
        # for i in range (-4,5):
        #     for j in range (-4,5):
        #         point = Point(head.x + 20*i, head.y - 20*j)
        #         state.append(game.is_collision(point))
        

        # state.append(game.food.x < game.head.x)
        # state.append(game.food.x > game.head.x)  # food right
        # state.append(game.food.y < game.head.y)  # food up
        # state.append(game.food.y > game.head.y)

        state = [     #state for the model with 11 values
            # Danger straight
            (direction_right and game.is_collision(point_right)) or 
            (direction_left and game.is_collision(point_left)) or 
            (direction_up and game.is_collision(point_up)) or 
            (direction_down and game.is_collision(point_down)),

            # Danger right
            (direction_up and game.is_collision(point_right)) or 
            (direction_down and game.is_collision(point_left)) or 
            (direction_left and game.is_collision(point_up)) or 
            (direction_right and game.is_collision(point_down)),

            # Danger left
            (direction_down and game.is_collision(point_right)) or 
            (direction_up and game.is_collision(point_left)) or 
            (direction_right and game.is_collision(point_up)) or 
            (direction_left and game.is_collision(point_down)),
            
            # Move direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

        
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)


    def get_action(self,state):
        self.epsilon = 80 - self.n_games/2 # the epsilon parameter allow us to move randomly in the begining 
        final_move = [0,0,0]               # to gather information 
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_pred = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_pred)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_test = []
    moove_mean = []
    total_score = 0
    record = 0
    agent = Agent()
    # agent.model.load()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward,done,score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        agent.remember(state_old,final_move,reward,state_new,done)
        if done:
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()
            if score>record:
                record = score
                agent.model.save("new_model.pth")

            print('game',agent.n_games,"score",score,"record",record )

            if len(moove_mean)<100:  #for the plots
                moove_mean.append(score)
                plot_test.append(sum(moove_mean)/len(moove_mean))
            else:
                moove_mean.append(score)
                moove_mean.pop(0)
                plot_test.append(sum(moove_mean)/len(moove_mean))
            

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            if agent.n_games%500 == 0:
                plot(plot_test)


if __name__ == '__main__':
    train()

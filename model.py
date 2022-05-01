from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def load(self, file_name):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print ('Loading existing model')
            return True
        print ('No model found.')
        return False

    def save(self,file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name) 

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.model = model
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()


    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        prediction = self.model(state)
        target = prediction.clone()
        for i in range(len(done)):
            QNew = reward[i]
            if not done[i]:
                QNew = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = QNew
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.loss(target, prediction)
        loss.backward()
        self.optimizer.step()


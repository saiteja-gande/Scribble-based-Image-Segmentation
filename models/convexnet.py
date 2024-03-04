import torch.nn as nn

# Define the fully connected neural network
class ConvexNet(nn.Module):
    def __init__(self,input_units,hidden_units):
        super(ConvexNet, self).__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.fc5 = nn.Linear(hidden_units, 1)
        self.skip1 = nn.Linear(input_units,hidden_units)
        self.skip2 = nn.Linear(input_units,hidden_units)
        self.skip3 = nn.Linear(input_units,hidden_units)
        self.skip4 = nn.Linear(input_units,1)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        p = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)+self.skip1(p)        
        x = self.relu(x)
        x = self.fc3(x)+self.skip2(p)
        x = self.relu(x)
        x = self.fc4(x)+self.skip3(p)
        x = self.relu(x)
        x = self.fc5(x)+self.skip4(p)
        x = self.Sigmoid(x)
        return x
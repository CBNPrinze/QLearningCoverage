import torch.nn as nn
import torch.nn.functional as F

# Define the Q-network architecture
class QNetwork(nn.Module):
    def __init__(self, input_channels, output_size,N):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1)
        

        # Correct the input size for the LSTM layer
        conv_output_size = self.calculate_conv_output_size(N)
        
        self.lstm_input_size = 16 * conv_output_size * conv_output_size

        self.lstm = nn.LSTMCell(self.lstm_input_size, 128)

        self.linear = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the output before passing it to the LSTM layer
        x = x.view(-1, 16 * 8 * 8)

        hx, cx = self.lstm(x)
        q_values = self.linear(hx)
        return q_values

    def calculate_conv_output_size(self, N):
        conv1_output_size = ((N - 5) // 1) + 1
        conv2_output_size = ((conv1_output_size - 5) // 1) + 1
        return conv2_output_size
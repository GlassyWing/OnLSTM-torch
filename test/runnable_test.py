import torch

from onlstm import OnLSTM

if __name__ == '__main__':
    on_lstm = OnLSTM(input_size=3,
                     hidden_size=12,
                     level_hidden_size=6,
                     num_layers=2,
                     bidirectional=False)
    inputs = torch.randn(2, 10, 3)
    outputs, (h, c) = on_lstm(inputs)
    print(outputs.shape)
    print(h.shape)
    print(c.shape)

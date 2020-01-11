import torch
import torch.nn as nn
from torch.nn import Parameter


class OnLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, level_hidden_size=None, bias=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.level_hidden_size = level_hidden_size
        self.n_repeat = hidden_size // level_hidden_size

        self.lstm_weight = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        self.level_weight = nn.Linear(input_size + hidden_size, 2 * level_hidden_size, bias=bias)

    def forward(self, input, prev_state):
        """

        :param input: shape of (bsz, 1, input_size)
        :param prev_state:  h,c from prev step witch shape of (1, hidden_dim)
        :return:
        """

        h_prev, c_prev = prev_state

        combined = torch.cat([input, h_prev], dim=-1)

        cc_i, cc_f, cc_o, cc_g = torch.split(self.lstm_weight(combined), self.hidden_size, dim=1)

        cc_i_h, cc_f_h = torch.split(self.level_weight(combined), self.level_hidden_size, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)

        c = torch.tanh(cc_g)

        p_f = torch.softmax(cc_f_h, dim=-1)
        p_i = torch.softmax(cc_i_h.flip(dims=[-1]), dim=-1)

        # level mask
        i_h = torch.cumsum(p_f, dim=-1)  # (1, level_hidden_size)
        f_h = torch.cumsum(p_i, dim=-1).flip(dims=[-1])  # (1, level_hidden_size)

        # (1, level_hidden_size, 1) -> (1, level_hidden_size, n_repeat) -> (1, hidden_size)
        i_h = i_h.unsqueeze(dim=-1).expand((*i_h.shape, self.n_repeat)).flatten(1)
        f_h = f_h.unsqueeze(dim=-1).expand((*f_h.shape, self.n_repeat)).flatten(1)

        w = i_h * f_h

        # combine information from lower and higher layer
        c = w * (f * c_prev + i * c) + (f_h - w) * c_prev + (i_h - w) * c
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(self, batch_size, dtype, device):
        state = (
            Parameter(torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device),
                      requires_grad=False),
            Parameter(torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device),
                      requires_grad=False)
        )
        return state


class OnLSTM(nn.Module):

    def __init__(self, input_size, hidden_size,
                 level_hidden_size=None,
                 num_layers=1, bias=True,
                 batch_first=True):
        super().__init__()

        assert num_layers >= 1, 'Need at least one layer'

        if level_hidden_size is None:
            level_hidden_size = hidden_size
        else:
            assert hidden_size % level_hidden_size == 0, \
                'level_hidden_size should be divisible by hidden_size'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_size if i == 0 else self.hidden_size
            cell_list.append(OnLSTMCell(input_size=cur_input_dim,
                                        hidden_size=hidden_size,
                                        level_hidden_size=level_hidden_size,
                                        bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state=None):
        """

        :param input: 3-D Tensor either of shape (b, t, d) or (t, b, d)
        :param hidden_state: come from existed hidden state or inited by zeros if None.
        :return:
        """

        if not self.batch_first:
            input = input.permute(1, 0, 2)

        if hidden_state is None:
            hidden_state = self._init_hidden(input.shape[0], input.dtype, input.device)

        last_state_list = []

        seq_len = input.size(1)
        cur_layer_input = input

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            last_state_list.append((h, c))

        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2)

        return layer_output, last_state_list

    def _init_hidden(self, batch_size, dtype, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, dtype, device))
        return init_states


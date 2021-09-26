import torch
import torch.nn as nn


def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.01)
        # torch.nn.init.normal_(m.weight.data)
        # torch.nn.init.normal_(m.bias.data)


class SigmoidNet(nn.Module):
    def __init__(self, hidden_neuron=2, num_layer=2):
        super(SigmoidNet, self).__init__()
        layer_lst = [nn.Linear(in_features=1,
                               out_features=hidden_neuron),
                     nn.Sigmoid()]
        for i in range(num_layer-2):
            layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                       out_features=hidden_neuron))
            layer_lst.append(nn.Sigmoid())
        layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                   out_features=1))
        layer_lst.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layer_lst)

    def forward(self, input_tensor):
        out = self.fc(input_tensor)
        return out


class ReLUNet(nn.Module):
    def __init__(self, hidden_neuron=2, num_layer=2,
                 activation="relu"):
        super(ReLUNet, self).__init__()
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "leaky":
            act_layer = nn.LeakyReLU()

        layer_lst = [nn.Linear(in_features=1,
                               out_features=hidden_neuron),
                     act_layer]
        for i in range(num_layer - 2):
            layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                       out_features=hidden_neuron))
            layer_lst.append(act_layer)
        layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                   out_features=1))
        layer_lst.append(act_layer)
        self.fc = nn.Sequential(*layer_lst)

    def forward(self, input_tensor):
        out = self.fc(input_tensor)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # test_model = SigmoidNet(hidden_num=2)
    test_model = ReLUNet(hidden_neuron=3, num_layer=4)
    test_model.apply(weights_init_normal)
    print("Weights Initialized.")

    test_input = torch.rand(size=[4, 1])
    test_output = test_model(test_input)
    print(test_output)

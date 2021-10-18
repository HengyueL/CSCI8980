import torch
import torch.nn as nn


def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        # torch.nn.init.normal_(m.bias.data, 0.0, 0.01)
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)


class Abs(nn.Module):
    """
        Custom Abs activation Layer.
    """
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        out = torch.abs(x)
        return out


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


class SigmoidNormNet(nn.Module):
    def __init__(self, hidden_neuron=2, num_layer=2):
        super(SigmoidNormNet, self).__init__()
        # norm_layer = nn.BatchNorm1d(num_features=hidden_neuron)
        # norm_layer = nn.LayerNorm(normalized_shape=hidden_neuron)

        layer_lst = [nn.Linear(in_features=1,
                               out_features=hidden_neuron),
                     # nn.BatchNorm1d(num_features=hidden_neuron),
                     nn.LayerNorm(normalized_shape=hidden_neuron),
                     nn.Sigmoid()]

        for i in range(num_layer-2):
            layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                       out_features=hidden_neuron))
            layer_lst.append(nn.LayerNorm(normalized_shape=hidden_neuron))
            # layer_lst.append(nn.BatchNorm1d(num_features=hidden_neuron))
            layer_lst.append(nn.Sigmoid())

        layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                   out_features=1))
        layer_lst.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layer_lst)

    def forward(self, input_tensor):
        out = self.fc(input_tensor)
        return out


class SigmoidFeatureNorm(nn.Module):
    def __init__(self, hidden_neuron=2, num_layer=2):
        super(SigmoidFeatureNorm, self).__init__()
        self.fc1 = nn.Sequential(*[nn.Linear(in_features=1,
                                    out_features=hidden_neuron)])
        assert num_layer == 3, "Did Not implement."
        self.fc2 = nn.Sequential(*[nn.Sigmoid(),
                                   nn.Linear(in_features=hidden_neuron,
                                             out_features=hidden_neuron)])

        self.fc3 = nn.Sequential(*[nn.Sigmoid(),
                                   nn.Linear(in_features=hidden_neuron,
                                             out_features=1)])
        self.act = nn.Sigmoid()

    def forward(self, input_tensor):
        out = input_tensor
        out = self.fc1(out)
        out = out / torch.norm(out)
        out = self.fc2(out)
        out = out / torch.norm(out)
        out = self.fc3(out)
        out = out / torch.norm(out)
        out = self.act(out)
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


class ReLUNormNet(nn.Module):
    def __init__(self, hidden_neuron=2, num_layer=2,
                 activation="relu"):
        super(ReLUNormNet, self).__init__()
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "leaky":
            act_layer = nn.LeakyReLU()
        # norm_layer = nn.BatchNorm1d(num_features=hidden_neuron)
        norm_layer = nn.LayerNorm(normalized_shape=hidden_neuron)
        layer_lst = [nn.Linear(in_features=1,
                               out_features=hidden_neuron),
                     act_layer,
                     norm_layer ]
        for i in range(num_layer - 2):
            layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                       out_features=hidden_neuron))
            layer_lst.append(act_layer)
            layer_lst.append(norm_layer)

        layer_lst.append(nn.Linear(in_features=hidden_neuron,
                                   out_features=1))
        layer_lst.append(act_layer)
        self.fc = nn.Sequential(*layer_lst)

    def forward(self, input_tensor):
        out = self.fc(input_tensor)
        return out


class AbsNet(nn.Module):
    def __init__(self, hidden_neuron=2, num_layer=2):
        super(AbsNet, self).__init__()
        act_layer = Abs()

        layer_lst = [nn.Linear(in_features=1,
                               out_features=hidden_neuron),
                     act_layer]

        for i in range(num_layer-2):
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

    test_model = AbsNet(hidden_neuron=4, num_layer=10)
    test_model.apply(weights_init_normal)
    print("Weights Initialized.")

    test_input = torch.rand(size=[4, 1])
    test_output = test_model(test_input)
    print(test_output)
    print()

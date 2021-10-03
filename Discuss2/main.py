import os.path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import SimpleRegression
from models import SigmoidNet, ReLUNet, AbsNet, weights_init_normal, SigmoidNormNet, ReLUNormNet
from torch.utils.data import DataLoader


def bump_func(x_array):
    """
        This function takes into a ndarray with shape (n,) as the 1-d function input x.
        The output is y, where y = f(x) element-wise, and f(.) is the bump fucntion defined in [0, 1]
    """
    assert np.amin(x_array) >= 0, "Input min value exceed the range."
    assert np.amax(x_array) <= 1, "Input max value exceed the range."
    y1 = np.where(x_array > 0.25, 1, 0)
    y2 = np.where(x_array < 0.75, 1, 0)
    y = np.multiply(y1, y2)
    return y


def train_net(model, dataloader,
              optimizer, lr_scheduler,
              total_epoch, save_dir, device):
    for epoch in range(total_epoch):
        epoch_loss = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()

            input_tensor = data["input"].to(device)
            label_tensor = data["label"].to(device)

            output_tensor = model(input_tensor)
            loss = loss_func(output_tensor, label_tensor)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("[Epoch {}] Total Training Loss:{}".format(epoch, epoch_loss))
        lr_scheduler.step()
    save_path = os.path.join(save_dir, "result.pth")
    torch.save(model.state_dict(),
               save_path)


def test_model(model, dataloader, device):
    input_lst, output_lst = [], []
    for i, data in enumerate(dataloader):
        input_tensor = data["input"].to(device)
        # label_tensor = data["label"].to(device)
        output_tensor = model(input_tensor)

        input_data = input_tensor.detach().data.cpu().numpy()
        output_data = output_tensor.detach().data.cpu().numpy()

        input_lst.append(input_data)
        output_lst.append(output_data)
        # print(input_data.shape)
        # print(output_data.shape)
    return np.concatenate(input_lst, axis=0), np.concatenate(output_lst, axis=0)


if __name__ == '__main__':
    # === Config for reproduction
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Exp setups
    total_epoch = 100
    num_data = 2000
    neurons = [24]
    layers = [4, 8, 16, 32]
    type_lst = ["SigmoidNorm", "LeakyNorm"]
    train_ids = [True, False]
    opt_type = "Adam"

    # === Visualization Folder Initialization
    save_fig_dir = os.path.join('./', "figures_%d_%s" % (num_data, opt_type))
    os.makedirs(save_fig_dir, exist_ok=True)
    save_model_dir = os.path.join('./', "%d_%s" % (num_data, opt_type))
    os.makedirs(save_model_dir, exist_ok=True)

    for net_type in type_lst:
        for num_layers in layers:
            for num_neurons in neurons:
                for is_train in train_ids:
                    # ===== Case one: small dataset, Sigmoid Network
                    if is_train:
                        x_1 = np.linspace(0, 1, num=num_data, endpoint=True)
                        y_1 = bump_func(x_1)
                        train_set_1 = SimpleRegression(x_1, y_1)
                        train_loader_1 = DataLoader(train_set_1, batch_size=64, shuffle=True,
                                                    num_workers=2)

                        # === Init a 1-hidden layer NN, Optimizer and Loss Function
                        if net_type == "ReLU":
                            net = ReLUNet(hidden_neuron=num_neurons,
                                          num_layer=num_layers)  # The visual proof network
                        elif net_type == "Sigmoid":
                            net = SigmoidNet(hidden_neuron=num_neurons,
                                             num_layer=num_layers)
                        elif net_type == "Leaky":
                            net = ReLUNet(hidden_neuron=num_neurons,
                                          num_layer=num_layers,
                                          activation="leaky")
                        elif net_type == "Abs":
                            net = AbsNet(hidden_neuron=num_neurons,
                                         num_layer=num_layers,)
                        elif net_type == "SigmoidNorm":
                            net = SigmoidNormNet(hidden_neuron=num_neurons,
                                                 num_layer=num_layers)
                        elif net_type == "LeakyNorm":
                            net = ReLUNormNet(hidden_neuron=num_neurons,
                                              num_layer=num_layers,
                                              activation="leaky")
                        else:
                            print("Undefined NN type. Check settings.")

                        net.apply(weights_init_normal)
                        if opt_type == "Adam":
                            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
                        elif opt_type == "SGD":
                            optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
                        else:
                            print("Need to specify the optimizer")
                        net.to(device)
                        net.train()

                        # Exponential Decay Learning rate
                        decayRate = 0.96
                        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
                        loss_func = torch.nn.MSELoss()

                        save_dir = os.path.join(save_model_dir, "%s_L%d_N%d" % (net_type,
                                                                                num_layers,
                                                                                num_neurons))
                        os.makedirs(save_dir, exist_ok=True)

                        train_net(model=net, dataloader=train_loader_1,
                                  optimizer=optimizer, lr_scheduler=my_lr_scheduler,
                                  total_epoch=total_epoch, save_dir=save_dir, device=device)
                    else:
                        # === Generate GT bump function benchmark
                        x_true = np.linspace(0, 1, num=3000, endpoint=True)
                        y_true = bump_func(x_true)
                        x_train = np.linspace(0, 1, num=num_data, endpoint=True)
                        y_train = bump_func(x_train)

                        test_set = SimpleRegression(x_true, y_true)
                        test_loader = DataLoader(test_set, batch_size=128, drop_last=True,
                                                 shuffle=False, num_workers=4)
                        if net_type == "ReLU":
                            net = ReLUNet(hidden_neuron=num_neurons,
                                          num_layer=num_layers)  # The visual proof network
                        elif net_type == "Sigmoid":
                            net = SigmoidNet(hidden_neuron=num_neurons,
                                             num_layer=num_layers)
                        elif net_type == "Leaky":
                            net = ReLUNet(hidden_neuron=num_neurons,
                                          num_layer=num_layers,
                                          activation="leaky")
                        else:
                            print("Undefined NN type. Check settings.")

                        save_dir = os.path.join(save_model_dir, "%s_L%d_N%d" % (net_type,
                                                                                num_layers,
                                                                                num_neurons))
                        model_weights_dir = os.path.join(save_dir, "result.pth")
                        net.load_state_dict(torch.load(model_weights_dir))

                        net.to(device)
                        net.eval()

                        test_input, test_output = test_model(net, test_loader,device=device)
                        print(test_input.shape)
                        print(test_output.shape)
                        # === Generate Plots
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
                        p1 = ax.plot(x_true, y_true, label='Bump_GT')
                        p2 = ax.scatter(x_train, y_train, label="Training Data Points")
                        p3 = ax.scatter(test_input[:, 0], test_output[:, 0], label="Network Prediction")
                        ax.grid()
                        ax.legend()
                        plt.savefig("%s/%s_L%d_N%d.jpg" % (save_fig_dir, net_type,
                                                           num_layers, num_neurons))
                        plt.close(fig)

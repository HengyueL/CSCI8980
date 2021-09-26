import os.path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import SimpleRegression
from models import SigmoidNet, ReLUNet, weights_init_normal
from torch.utils.data import DataLoader
from main import test_model, bump_func


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Generate Sigmoid Result
    net_type = "Sigmoid"  # "Leaky", "Sigmoid" or "ReLU" are valid input
    num_data = 2000
    neurons = [4, 12, 24, 48, 64]
    # layers = [2, 4, 8]
    layers = [8]

    # === Visualization Folder Initialization
    save_model_dir = os.path.join('./', "%d" % num_data)

    save_fig_dir = os.path.join('./', "final_figs_%d" % num_data)
    os.makedirs(save_fig_dir, exist_ok=True)

    # === Generate Samples to Plot
    x_true = np.linspace(0, 1, num=1500, endpoint=True)
    y_true = bump_func(x_true)

    # === Generate Plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    p1 = ax.plot(x_true, y_true, label='Bump_GT')

    test_set = SimpleRegression(x_true, y_true)
    test_loader = DataLoader(test_set, batch_size=128, drop_last=True,
                             shuffle=False, num_workers=4)
    for num_neurons in neurons:
        for num_layers in layers:
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
                print("Something Wrong.")
            # Load Pretrained Model and Compute
            save_dir = os.path.join(save_model_dir, "%s_L%d_N%d" % (net_type,
                                                                    num_layers,
                                                                    num_neurons))
            model_weights_dir = os.path.join(save_dir, "result.pth")
            net.load_state_dict(torch.load(model_weights_dir))
            net.to(device)
            net.eval()
            test_input, test_output = test_model(net, test_loader, device=device)

            # Add to plot
            p2 = ax.scatter(test_input[:, 0], test_output[:, 0],
                            label="L%d_N%d" % (num_layers, num_neurons))
    ax.set_title("%s with %d Layers" % (net_type, layers[0]))
    ax.grid()
    ax.legend()
    plt.savefig("%s/%s_L%d.jpg" % (save_fig_dir, net_type, layers[0]))
    plt.close(fig)

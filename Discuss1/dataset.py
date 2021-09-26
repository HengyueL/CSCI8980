import torch
import numpy as np
from torch.utils.data import Dataset


class SimpleRegression(Dataset):
    """
        This dataset takes ndarray as input.
        For training convenience.
    """
    def __init__(self, x, y):
        super(SimpleRegression, self).__init__()
        self.input_data = x
        self.label = y

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, item):
        input_data = np.asarray([self.input_data[item]])
        label_data = np.asarray([self.label[item]])
        res = {
            "input": torch.as_tensor(input_data).float(),
            "label": torch.as_tensor(label_data).float()
        }
        return res


if __name__ == "__main__":
    x = np.random.random(size=(4,))
    y = np.random.random(size=(4,))
    print(x)
    print(y)

    test_dataset = SimpleRegression(x, y)
    test_sample = test_dataset[0]

    print(test_sample["input"])
    print(test_sample["label"])


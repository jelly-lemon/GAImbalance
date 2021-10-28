import numpy as np


def convert(data_name, label):
    label_nums = {
        "Abalone": {b"M": 0, b"F": 1, b"I": 2},
        "Car": {b"unacc": 0, b"acc": 1, b"good": 2, b"vgood": 3},
    }
    return label_nums[data_name][label]


def get_data(data_name):
    x = y = None
    path = "./data/"
    if data_name == "Abalone":
        path += "Abalone.data"
        data = np.loadtxt(path, dtype=float, delimiter=',', converters={0: lambda label: convert("Abalone", label)})
        y, x = np.split(data, indices_or_sections=[1, ], axis=1)
        y = y.reshape(y.shape[0])
    elif data_name == "Car":
        path += "Car.data"
        data = np.loadtxt(path, dtype=str, delimiter=',')
        new_data = []
        convert_dict = {
            "buying": {"vhigh": 0, "high": 1, "med": 2, "low": 3},
            "maint": {"vhigh": 0, "high": 1, "med": 2, "low": 3},
            "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
            "persons": {"2": 0, "4": 1, "more": 2},
            "lug_boot": {"small": 0, "med": 1, "big": 2},
            "safety": {"low": 0, "med": 1, "high": 2},
            "label": {"unacc":0, "acc":1, "good":2, "vgood":3}
        }
        for d1 in data:
            new_d1 = [convert_dict["buying"][d1[0]], convert_dict["maint"][d1[1]],
                      convert_dict["doors"][d1[2]], convert_dict["persons"][d1[3]],
                      convert_dict["lug_boot"][d1[4]], convert_dict["safety"][d1[5]],
                      convert_dict["label"][d1[6]]]
            new_data.append(new_d1)
        x, y = np.split(new_data, indices_or_sections=[6, ], axis=1)
        y = y.reshape(y.shape[0])
    return x, y


def split(data, val_ratio):
    x_train = y_train = x_val = y_val = None
    for key in data:
        np.random.shuffle(data[key])
        n = len(data[key])
        n_train = int(n * (1 - val_ratio))
        if x_train is None:
            x_train = np.array(data[key][:n_train])
            y_train = np.full(n_train, key, dtype="float32")
            x_val = np.array(data[key][n_train:])
            y_val = np.full(n - n_train, key, dtype="float32")
        else:
            x_train = np.vstack((x_train, data[key][:n_train]))
            y1 = np.full(n_train, key, dtype="float32")
            y_train = np.concatenate((y_train, y1))
            x_val = np.vstack((x_val, data[key][n_train:]))
            y2 = np.full(n - n_train, key, dtype="float32")
            y_val = np.concatenate((y_val, y2))

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    get_data("Car")

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def show_result(x, y_dict, model_name=''):
    fig, ax = plt.subplots()

    for key,val in y_dict.items():
        ax.plot(x, np.array(val)*100, label=key)

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy(%)")
    ax.legend()
    ax.grid()
    ax.set_title("Actvation function comparision ({})".format(model_name))
    plt.show()

#EEGNet_lr0.001_ep1000.json
#DeepConvNet_lr0.001_ep300.json

def main():
    source = {}
    with open("DeepConvNet_lr0.001_ep300.json", 'r') as f:
        source = json.load(f)
    pprint(list(zip(source['y_dict'].keys(), [max(i) for i in source['y_dict'].values()])))
    show_result(source['x'], source['y_dict'], source['title'])

if __name__ == "__main__":
    main()
    #sys.stdout.flush()
    #sys.exit()

#%%
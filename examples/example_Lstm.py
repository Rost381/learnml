import numpy as np

from learnml.deep.api import LstmNetwork, LstmParam
from learnml.deep.layers import SimpleLstmLoss


def main():
    np.random.seed(0)
    mem_cell_ct = 100
    x_dim = 50
    lp = LstmParam(mem_cell_ct, x_dim)
    ln = LstmNetwork(lp)
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):
            ln.x_list_add(input_val_arr[ind])

        print("y_pred = [" + ", ".join(["% 2.5f" % ln.lstm_node_list[ind].state.h[0]
                                        for ind in range(len(y_list))]) + "]", end=", ")

        loss = ln.y_list_is(y_list, SimpleLstmLoss)
        print("loss:", "%.3e" % loss)
        lp.apply_diff(lr=0.1)
        ln.x_list_clear()


if __name__ == "__main__":
    main()

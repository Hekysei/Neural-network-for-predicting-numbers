import pandas as pd
import numpy as np

# Power of two - size of the predicted number
power = 7
# Prediction error - 1/mult
mult = 2
# Number of hidden neurons
hid = 30
# How much the neural network will change with each backpropagation
learning_rate = 0.02
# Control training with zero error
control_training = 6000


delta_num = 8*(2**mult)
inp = power+mult
out = power+mult

def to_bin(n):
    n *= 2**mult
    n += delta_num
    res = bin(round(n))[2:]
    while len(res) < inp:
        res = "0"+res
    return res

def unbin(ls):
    ls = [str(round(x[0])) for x in ls]
    res = int("".join(ls), 2) - delta_num
    res /= 2**mult
    return res

def get_learned():
    # Put output of learning here
    weights_input_to_hidden = np.array(...)
    weights_hidden_to_output = np.array(...)
    bias_input_to_hidden = np.array(...)
    bias_hidden_to_output = np.array(...)

    return weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output


def nn_work(weights_input_to_hidden,
            weights_hidden_to_output,
            bias_input_to_hidden,
            bias_hidden_to_output,
            q):
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ q
    hidden = 1 / (1 + np.exp(-hidden_raw))

    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    return output, hidden


def learn_nn(df):
    weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hid, inp))
    weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (out, hid))
    bias_input_to_hidden = np.zeros((hid, 1))
    bias_hidden_to_output = np.zeros((out, 1))

    control_count = 0
    epoch = 0

    while control_count != control_training:
        lose = 0
        epoch += 1

        print(f"Epoch №{epoch}; cc: {control_count}")

        # Take a random sample of indexes
        rows = np.random.choice(len(df), size=15)

        for row in df[rows]:
            q = np.array([float(x) for x in row[0]]).reshape((-1, 1))
            a = np.array([float(x) for x in row[1]]).reshape((-1, 1))

            output, hidden = nn_work(weights_input_to_hidden,
                                     weights_hidden_to_output,
                                     bias_input_to_hidden,
                                     bias_hidden_to_output,
                                     q)
            if unbin(a) != unbin(output):
                lose += 1

            # Backpropagation magic
            delta_output = output - a
            weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
            bias_hidden_to_output += -learning_rate * delta_output

            delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
            weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(q)
            bias_input_to_hidden += -learning_rate * delta_hidden

        if lose == 0:
            control_count += 1
        else:
            control_count = 0

        print(f'Lose: {lose}\n')

    print("Copy the output below and put it in the get_learned() function\n")
    print(f'weights_input_to_hidden = np.array({weights_input_to_hidden.tolist()})')
    print(f'weights_hidden_to_output = np.array({weights_hidden_to_output.tolist()})')
    print(f'bias_input_to_hidden = np.array({bias_input_to_hidden.tolist()})')
    print(f'bias_hidden_to_output = np.array({bias_hidden_to_output.tolist()})\n')

    return weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output


def process(df):
    df = df.dropna()
    df["x"] = np.vectorize(to_bin)(df["x"])
    df["y"] = np.vectorize(to_bin)(df["y"])
    df = df.to_numpy()

    # If you have already trained the model, use get_learned(), otherwise learn_nn(df)
    nn = learn_nn(df)
    # nn = get_learned()

    df = {"x": [], "pred": []}
    for x in range(75, 100):
        q = np.array([float(x) for x in to_bin(x)]).reshape((-1, 1))
        output, _ = nn_work(*nn, q)
        df["x"].append(x)
        df["pred"].append(unbin(output))

    df = pd.DataFrame(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv('ts_prediction.csv')
    print(process(df))
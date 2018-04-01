from algorithm import *

# example 1
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

test_row = dataset[0]

r = bmu(dataset, test_row)

print(r)

print(random_codebook(dataset))

n_codebooks = 2
lrate = 0.3
epochs = 10

codebooks = train_codebooks(dataset, n_codebooks, lrate, epochs)

print('Codebooks: %s' % codebooks)

for i in dataset:
    print(predict(codebooks, i))
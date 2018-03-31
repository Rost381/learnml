dataset = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

r = [c for c in zip(*dataset)]

print(r)

dataset = [
    ['a', 0],
    ['b', 100],
    ['c', 26]
]

dataset.sort(key=lambda x: x[1])

print(dataset)

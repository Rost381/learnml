# Pilot

Important tricks for ml.

## Get column as list 
```python
dataset = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

r = [c for c in zip(*dataset)]
# [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
```

## Sort by column
```python
dataset = [
    ['a', 0],
    ['b', 100],
    ['c', 26]
]

dataset.sort(key=lambda x: x[1])
# [['a', 0], ['c', 26], ['b', 100]]
```

## Get index of max value in list
```python
x = [0.1, 0.9]
print(x.index(max(x))) # 1

y = [0.9, 0.1]
print(y.index(max(y))) # 0
```
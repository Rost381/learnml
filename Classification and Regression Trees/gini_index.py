def gini_index(data_groups, class_values):

    r = 0.0

    for c in class_values:
        for g in data_groups:
            l = len(g)
            if l == 0:
                continue

            prop = [row[-1] for row in g].count(c) / float(l)
            print('prop: %s' % prop)
            r += (prop * (1.0 - prop))

    return r
"""
prop:
0,1/2,1/2
1/2,1/2,0
1/2,0,1/2 

prop*(1-prop)
0,1/4,1/4
1/4,1/4,0
1/4,0,1/4

sum = 3/2
"""

#print(gini_index([[[1, 1], [1, 2]], [[1, 1], [1, 0]], [[1, 2], [1, 0]]], [0, 1, 2]))

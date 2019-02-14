class A:
    
    def __init__(self):
        print("A-init")

    def __call__(self):
        print("A-call")

    def fit(self, x):
        print("A-fit")
        return x + 1


class B(A):
    def fit(self, x):
        print("B-fit")
        return x + 2
        super(B, self).fit()


"""
__init__ used to initialise
__call__ implements function call operator.
"""
a = A()  # A-init
a()  # A-call

"""
super(B, self).fit() replace the father function
"""
b = B()  # A-init
b.fit(1)  # B-fit
print(b.fit(1))  # 3

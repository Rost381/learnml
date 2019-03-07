class A:
    def __init__(self):
        print("A-init")

    def __call__(self):
        print("A-call")

    def fit(self, x):
        print("A-fit")
        return x + 1


class B(A):
    def __init__(self):
        self.x = 'x'
        self._y = 'x'
        self.__z = 'z'

    def fit(self, x):
        print("B-fit")
        return x + 2
        super(B, self).fit()


class C(A):
    def __init__(self):
        self.child_attribute = 'child'


"""
__init__ used to initialise
__call__ implements function call operator.
"""
a = A()  # A-init
a()  # A-call
a.fit(1)
"""
super(B, self).fit() replace the father function
"""
b = B()  # A-init
b.fit(1)  # B-fit
print(b.fit(1))  # 3

attrs = vars(b)

print(', '.join("%s: %s" % item for item in attrs.items())) # x: x, _y: x, _B__z: z
print(attrs['_B__z'])  # z

c = A()
attrs = vars(c)
print(', '.join("%s: %s" % item for item in attrs.items())) # A-init

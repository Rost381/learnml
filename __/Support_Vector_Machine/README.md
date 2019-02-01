## Optimization problem

$$
\max _{\alpha }\sum _{i=1}^{n}\alpha _{i}-{\frac {1}{2}}\sum _{i=1}^{n}\sum _{j=1}^{n}y_{i}y_{j}K(x_{i},x_{j})\alpha _{i}\alpha _{j},
$$

subject to:

$$
L=max(0,a_2-a_1),\quad H=min(C,C+a_2-a_1)
$$

$$
L=max(0,a_2+a_1-C),\quad H=min(C,a_2+a_2)
$$

$$
\leq \alpha _{i}\leq C,\quad i=1,2,\ldots ,n,
$$

$$
{\displaystyle \sum _{i=1}^{n}y_{i}\alpha _{i}=0}
$$

$$
a_2^{new}=a_2+\frac{y_2(E_1-E_2)}{\eta}
$$

$$
a_1^{\text{new}}=a_1+s(a_2-a_2^{\text{new,clip}})
$$

$$
\eta=K(x_1,x_1)+K(x_2,x_2)-2K(x_1,x_2)
$$

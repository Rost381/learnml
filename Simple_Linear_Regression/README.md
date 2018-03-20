# Simple Linear Regression
simple linear regression is a linear regression model with a single explanatory variable.

## model function
$$
{\displaystyle y=\alpha +\beta x}
$$

## residuals
$$
{\displaystyle {\hat {\varepsilon }}_{i}=y_{i}-a-bx_{i}.}
$$

## α̂ and β̂ solve the following minimization problem
$$
{\displaystyle {\text{Find }}\min _{a,\,b}Q(a,b),\quad {\text{for }}Q(a,b)=\sum _{i=1}^{n}{\hat {\varepsilon }}_{i}^{\,2}=\sum _{i=1}^{n}(y_{i}-a-bx_{i})^{2}\ .}
$$

## values of a and b

$$
{\displaystyle {\begin{aligned}{\hat {\alpha }}&={\bar {y}}-{\hat {\beta }}\,{\bar {x}}\end{aligned}}}
$$

$$
{\displaystyle {\begin{aligned}\\{\hat {\beta }}&={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}\\\end{aligned}}}
$$


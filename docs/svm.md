# Support Vector Machines

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

Quadratic Programming with Python and CVXOPT

$$
\begin{array} { c } { \min _ { x } \frac { 1 } { 2 } x ^ { \top } P x + q ^ { \top } x } \\ { \text { subject to } \quad G x \preceq h } \\ { A x = b } \end{array}
$$

Consider a simple example:

$$
\begin{array}{c}{\min _{x, y} \frac{1}{2} x^{2}+3 x+4 y} \\ {\text { subject to } \quad x, y \geq 0} \\ {x+3 y \geq 15} \\ {2 x+5 y \leq 100} \\ {3 x+4 y \leq 80}\end{array}
$$

We rewrite the above in the given standard form:

$$
\min _{x, y} \frac{1}{2} \left[ \begin{array}{l}{x} \\ {y}\end{array}\right]^{\top} \left[ \begin{array}{ll}{1} & {0} \\ {0} & {0}\end{array}\right] \left[ \begin{array}{l}{x} \\ {y}\end{array}\right]+\left[ \begin{array}{l}{3} \\ {4}\end{array}\right]^{\top} \left[ \begin{array}{l}{x} \\ {y}\end{array}\right]
$$

$$
\left[ \begin{array}{cc}{-1} & {0} \\ {0} & {-1} \\ {-1} & {-3} \\ {2} & {5} \\ {3} & {4}\end{array}\right] \left[ \begin{array}{c}{x} \\ {y}\end{array}\right] \leq \left[ \begin{array}{c}{0} \\ {0} \\ {-15} \\ {100} \\ {80}\end{array}\right]
$$
## Stochastic Gradient Descent

Gradient Descent is the process of minimizing a function following the slope or gradient of that function.

* Could compute gradient over entire data set on each step, 1. but this turns out to be unnecessary
* Computing gradient on small data samples works well
    * On every step, get a new random sample
* Stochastic Gradient Descent: one example at a time
* Mini-Batch Gradient Descent: batches of 10-1000
    * Loss & gradients are averaged over the batch
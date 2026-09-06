# DNNS.jl
A Julia package to do Deep Learning.
It consists of 4 modules in the **src** directory:
- AutoDiff
    This is essentially the implementation of a *dual* number system along with
    how to evaluate them with basic arithmatic, log, exponential, and trigonometric functions.
    Linear combinations of functions are supported -- so automatically we can also evaluate dual numbers on
    polynomials.
    Finally, the chain rule as well as matrix/vector multiplication is supported.
- PWLF
    An implementation of piece-wise linear functions. This includes their values on *dual* numbers.
- UtilFunc
    An implementation of non-linear functions used when computing neural nets. Again, this includes their 
    values on *dual* numbers.
- DNNS
    The top level module used to construct and fit Deep Neural Networks.
    The main two methods provided are `loss` and `fit` (a fitted `DNN` is itself callable on an input vector).
    The `fit` function performs a simple full-batch, coordinate-wise gradient descent:
    on every iteration it loops through each parameter of each layer, computes the associated
    partial derivative of the loss over the whole data set using the `AutoDiff` dual numbers,
    and uses that value to "descend" a small amount. There is no stochastic batching.

A Jupyter notebook is also provided demonstrating the DNNS module.
It creates a simple network to fit a noisy straight line.

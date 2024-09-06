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
    The main two methods provided are `fit` and `predict`.
    The `fit` function uses a simplistic kind of stochastic batching to achieve 
    the gradient descent. It effectively computes the partial derivative for each parameter.
    Specifically, the `fit` function loops through each parameter, computes the associated partial
    derivative, and uses that as the gradient to "descend" a small amount.

A Jupyter notebook is also provided demonstrating the DNNS module.
It creates a simple network to fit a noisy straight line.

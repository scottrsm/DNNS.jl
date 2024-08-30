# DNNS.jl
A Julia package to do Deep Learning.
It consists of 4 modules:
- AutoDiff
    This is essentially the implementation of a *dual* number system along with
    how to evaluate them with basic arithmatic, log, exponential, and trigonometric functions.
    Linear combinations of functions are supported -- so automatically we can also evaluate dual numbers on
    polynomials.
    Finally, the chain rule as well as matrix/vector multiplication is supported.
- PWLF
    An implementation of piece-wise liner functions. This includes their values on *dual* numbers.
- UtilFunc
    An implementation of non-linear functions used when computing neural nets. Again, this includes their 
    values on *dual* numbers.
- DNNS
    The top level module used to construct and fit Deep Neural Networks.


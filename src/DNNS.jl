module DNNS

include("AutoDiff.jl")
import .AutoDiff: AD, _ad_matvec, _ad_dot

include("UtilFunc.jl")
using .UtilFunc

include("PWLF.jl")
using .PWLF

export AD, PWL, smooth
export sigmoid1, sigmoid2, sigmoid3, relu, relur, L1, softmax
export DLayer, DNN, loss, fit



"""
    DLayer{T<:Number, F}

A structure representing one layer of a neural net. 

## Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.
- `F` is the type of the activation function (any callable).

## Fields
- `M    :: Matrix{AD{T}}`    -- The weight matrix.
- `b    :: Vector{AD{T}}`    -- The bias vector.
- `op   :: F`                -- The activation (threshold) function, applied element-wise.
- `dims :: Tuple{Int, Int}`  -- The (output, input) dimensions.
                         

## Public Constructors
`DLayer(Mn::AbstractMatrix{<:Number}, bn::AbstractVector{<:Number}, opn)`
- `Mn` -- A `MxN` weight matrix (`N` inputs, `M` outputs).
- `bn` -- An `M` dimensional bias vector.
- `op` -- The non-linear threshold function.
The element type `T` is the promotion of the element types of `Mn` and `bn`;
`DLayer{T}(Mn, bn, opn)` converts both to `T`.
"""
struct DLayer{T<:Number, F}
    M::Matrix{AD{T}}
    b::Vector{AD{T}}
    op::F
    dims::Tuple{Int,Int}

    function DLayer{T}(Mn::AbstractMatrix{<:Number}, bn::AbstractVector{<:Number}, opn::F) where {T<:Number, F}
        n, m = size(Mn)
		length(bn) == n || throw(DomainError(length(bn), "DLayer (Inner Constructor): Matrix, `Mn` ($(size(Mn))), and vector, `bn` ($(length(bn))), are incompatible."))

        return new{T, F}(AD{T}.(Mn), AD{T}.(bn), opn, (n, m))
    end
end

# Outer constructor
DLayer(Mn::AbstractMatrix{S}, bn::AbstractVector{U}, opn) where {S<:Number, U<:Number} = DLayer{promote_type(S, U)}(Mn, bn, opn)



"""
	(L::DLayer{T})(x::AbstractVector) where {T <: Number}
If `(N,M) = L.dims`, then we may treat the structure, `DLayer`, as
a function: ``{\\cal R}^M \\mapsto {\\cal R}^N`` .

Takes input `x` and passes it through the layer.

# Type Constraints
- `T <: Number`

# Arguments
- `x :: AbstractVector`  -- An input vector of dimension `M`.

# Return
`::Vector{AD{T}}` of dimension `N`.
"""
function (L::DLayer{T})(x::AbstractVector) where {T<:Number}
	length(x) == L.dims[2] || throw(DomainError(length(x), "DLayer (As Function): Vector `x` ($(length(x))) is incompatible with layer dimensions ($(L.dims))."))

    return L.op.(_ad_matvec(L.M, x, AD{T}) .+ L.b)
end



"""
    DNN{T<:Number}

A structure representing a neural network. 

## Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.

## Fields
- `layers :: Vector{DLayer{T}}` -- The neural net layers.
                         

## Public Constructors
`DNN(ls::AbstractVector{<:DLayer{T}})`
- `ls` -- A vector of DLayer (all with the same element type `T`).
"""
struct DNN{T<:Number}
    layers::Vector{DLayer{T}}

    function DNN{T}(ls::AbstractVector{<:DLayer{T}}) where {T<:Number}
		length(ls) != 0 || throw(DomainError(length(ls), "DNN (Inner Constructor): Length of ls is 0."))
        for i in 1:(length(ls) - 1)
			ls[i].dims[1] == ls[i+1].dims[2] || throw(DomainError((ls[i].dims, ls[i+1].dims), "DNN (Inner Constructor): DLayer incompatibility between layers $i and $(i+1)."))
        end
        return new{T}(collect(DLayer{T}, ls))
    end
end

# Outer Constructor
DNN(ls::AbstractVector{<:DLayer{T}}) where {T<:Number} = DNN{T}(ls)


"""
	(dnn::DNN{T})(x::AbstractVector) where {T <: Number}
Let `N = DNN.ls[1].dims[2]` and `M = DNN.ls[end].dims[1]`, then
here we treat the structure `DNN` as a function: ``{\\cal R}^N \\mapsto {\\cal R}^M``
Takes input `x` and passes it through each of the layers of `DNN`.

# Type Constraints
- `T <: Number`

# Arguments
- `x :: AbstractVector`  -- An input vector of dimension `N` (numbers, or `AD` values).

# Return
`::Vector{AD{T}}` of dimension `M`.
"""
function (dnn::DNN{T})(x::AbstractVector) where {T<:Number}

    _, n = size(dnn.layers[1].M)
	length(x) == n || throw(DomainError(length(x), "DNN (as function): Matrix from first layer ($(dnn.layers[1].dims)) is incompatible with `x` ($(length(x)))."))

    y = dnn.layers[1](x)
    for i in 2:length(dnn.layers)
        y = dnn.layers[i](y)
    end

    return y
end


# Treat all parameters of the layer as constants (zero derivative).
function make_const!(l::DLayer{T}) where {T<:Number}
    t0 = zero(T)
    @inbounds for i in eachindex(l.b)
        l.b[i] = AD{T}(l.b[i].v, t0)
    end

    @inbounds for i in eachindex(l.M)
        l.M[i] = AD{T}(l.M[i].v, t0)
    end

    return nothing
end

# Set the derivative of the `k`th bias entry.
function set_bd_pd!(l::DLayer{T}, k::Int, d::T) where {T<:Number}
    l.b[k] = AD{T}(l.b[k].v, d)

    return nothing
end

# Set the derivative of the `k`th weight entry (linear index).
function set_md_pd!(l::DLayer{T}, k::Int, d::T) where {T<:Number}
    l.M[k] = AD{T}(l.M[k].v, d)

    return nothing
end


"""
    loss(dnn, X, Y)

Computes the loss of the neural network given inputs, `X`, and outputs `Y`.

# Type Constraints
- `T <: Number`

# Arguments
- `dnn :: DNN{T}`             -- A DNN.
- `X   :: AbstractMatrix`     -- The matrix of input values (one sample per column).
- `Y   :: AbstractMatrix`     -- The matrix of output values (one sample per column).

# Return
`::AD{T}` -- The (mean squared) loss of the network.
"""
function loss(dnn::DNN{T}, X::AbstractMatrix{<:Number}, Y::AbstractMatrix{<:Number}) where {T<:Number}
    _, m = size(X)
    _, my = size(Y)
	m == my || throw(DomainError((size(X), size(Y)), "`loss`: Dimensions of `X` and `Y` are incompatible."))
	m > 0 || throw(DomainError(m, "`loss`: There must be at least one sample."))
	size(Y, 1) == dnn.layers[end].dims[1] || throw(DomainError(size(Y, 1), "`loss`: The rows of `Y` do not match the output dimension of the network ($(dnn.layers[end].dims[1]))."))

    s = zero(AD{T})
    @inbounds for i in 1:m
        df = dnn(@view X[:, i]) .- (@view Y[:, i])
        s += _ad_dot(df, df, AD{T})
    end

    return s / T(m)
end


"""
    fit(dnn, X, Y; N=1000, relerr=1.0e-6, μ=1.0e-3, verbose=false)

Adjusts the parameters of neural network, `dnn`, **in place** to get the best fit of 
the data: `X`, `Y` (gradient descent on the loss, with the gradient computed by
forward mode automatic differentiation, one parameter at a time).
The parameters of the network are all pairs of 
matrices and biases for each layer in the network.

# Type Constraints
- `T <: Number`

# Arguments
- `dnn :: DNN{T}`         -- The network to fit (modified).
- `X   :: AbstractMatrix` -- The matrix of input values (one sample per column).
- `Y   :: AbstractMatrix` -- The matrix of output values (one sample per column).

# Keyword Arguments
- `N::Int=1000`         -- The maximum number of iterations.
- `relerr::Real=1.0e-6` -- Stop (after 20 iterations) once the relative change of the loss is at most `relerr`.
- `μ::Real=1.0e-3`      -- The learning rate.
- `verbose::Bool=false` -- If `true`, print the loss at each iteration and a summary at the end.

# Return
A named tuple `(loss, iterations, converged)`: the final loss, the number of
iterations used, and whether the relative-change stopping criterion was met.
"""
function fit(dnn::DNN{T}, X::AbstractMatrix{<:Number}, Y::AbstractMatrix{<:Number};
		N::Int=1000, relerr::Real=1.0e-6, μ::Real=1.0e-3, verbose::Bool=false) where {T<:Number}

    _, m = size(X)
    _, my = size(Y)

	m == my || throw(DomainError((size(X), size(Y)), "`fit`: Arrays, `X`, and `Y`, are incompatible."))
	N >= 0  || throw(DomainError(N, "`fit`: The number of iterations must be non-negative."))

	lss::T = loss(dnn, X, Y).v
	lss_last::T = lss
	rel_chg::T = typemax(T)
    finished_early = false
    num_iterates::Int = N
    mu::T = T(μ)
    relerr_t::T = T(relerr)
    @inbounds for j in 1:N
        rel_chg = lss_last == zero(T) ? zero(T) : abs((lss - lss_last) / lss_last)
        if j > 20 && rel_chg <= relerr_t && lss <= lss_last
            finished_early = true
            num_iterates = j - 1
            break
        end
        verbose && println("Iteration $(j): loss = $lss")
        lss_last = lss
        # Walk over each layer...
        for i in eachindex(dnn.layers)
			brat = one(T) * length(dnn.layers[i].M) / length(dnn.layers[i].b)
            # Treat the M and b parameters for this layer as constants.
            make_const!(dnn.layers[i])

            # Selectively treat the kth element of M as a variable so that
            # we may take the partial derivative with respect to M[k].
            for k in eachindex(dnn.layers[i].M)
               	set_md_pd!(dnn.layers[i], k, one(T))
               	ls = loss(dnn, X, Y)
               	set_md_pd!(dnn.layers[i], k, zero(T))
               	dnn.layers[i].M[k] = AD{T}(dnn.layers[i].M[k].v - ls.d * mu, zero(T))
            end

            # Selectively treat the kth element of b as a variable so that
            # we may take the partial derivative with respect to b[k].
            for k in eachindex(dnn.layers[i].b)
               	set_bd_pd!(dnn.layers[i], k, one(T))
               	ls = loss(dnn, X, Y)
               	set_bd_pd!(dnn.layers[i], k, zero(T))
                dnn.layers[i].b[k] = AD{T}(dnn.layers[i].b[k].v - ls.d * brat * mu, zero(T))
            end
        end
        lss = loss(dnn, X, Y).v
    end
    if verbose
        if finished_early
            println("Total number of iterates tried = $num_iterates from a max of $N.")
        else
            println("Used the maximum number of iterates ($N).")
        end
        println("The relative change of the loss = $rel_chg.")
    end

    return (loss=lss, iterations=num_iterates, converged=finished_early)
end

end # DNNS module


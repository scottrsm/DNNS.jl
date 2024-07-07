module DNNS

include("UtilFunc.jl")
using .UtilFunc

include("PWLF.jl")
using .PWLF

export AD, PWL, sigmoid1, sigmoid2, sigmoid3, relu, relur, L1, softmax
export DLayer, DNN, loss, fit

import StatsBase: sample
import LinearAlgebra as LA
import LinearAlgebra: dot


"""
    DLayer{T<:Number}

A structure representing one layer of a neural net. 

## Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.

## Fields
- `M    :: Matrix{AD{T}}`    -- The "x" values.
- `b    :: Vector{AD{T}}`    -- The "y" values.
- `op   :: Function`         -- The "slopes" of the segments.
- `dims :: Tuple{Int, Int}`  -- The number of "x,y" values.
                         

## Public Constructors
`DLayer(Mn::Matrix{T}, bn::Vector{T}, opn::Function)`
- `Mn` -- A `MxN`weight matrix.
- `bn` -- A `N`dimensional bias vector.
- `op` -- The non-linear threshold function.
"""
struct DLayer{T<:Number}
    M::Matrix{AD{T}}
    b::Vector{AD{T}}
    op::Function
    dims::Tuple{Int,Int}

    function DLayer{T}(Mn::Matrix{T}, bn::Vector{T}, opn::Function) where {T<:Number}
        n, m = size(Mn)
		length(bn) == n || throw(DomainError(m, "DLayer (Inner Constructor): Matrix, `Mn`, and vector, `bn`, are incompatible."))

        return new{T}(AD{T}.(Mn), AD{T}.(bn), opn, (n, m))
    end
end

# Outer constructor
DLayer(Mn::Matrix{T}, bn::Vector{T}, opn) where {T<:Number} = DLayer{T}(Mn, bn, opn)



"""
	(L::DLayer{T})(x::AD{T}) where {T <: Number}
If `(M,N) = L.dims`, then we may treat the structure, `DLayer`, as
a function: ``{\\cal R}^m \\mapsto {\\cal R}^n`` .

Takes input `x` and passes it through the layer.

# Type Constraints
- `T <: Number`

# Arguments
- `x :: AD{T}`  -- An input value of dimension `N`.

# Return
`::Vector{AD{T}}` of dimension `M`.
"""
function (L::DLayer{T})(x::AD{T}) where {T<:Number}
	length(x) == L.dims[1] || throw(DomainError(L.dims, "DLayer (As Function): Vector `x` is incompatible with layer dimensions."))

    return L.op.(L.M * x .+ L.b)
end



"""
    DNN{T<:Number}

A structure representing one layer of a neural net. 

## Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.

## Fields
- `layers :: Vector{DLayer{T}` -- The neural net layers.
                         

## Public Constructors
`DNN(ls::Vector{DLayer{T}}`
- `ls` -- A vector of DLayer.
"""
struct DNN{T<:Number}
    layers::Vector{DLayer{T}}

    function DNN{T}(ls::Vector{DLayer{T}}) where {T<:Number}
		length(ls) != 0 || throw(DomainError(length(ls), "DNN (Inner Constructor): Length of ls is 0."))
        for i in eachindex(ls[1:end-1])
			ls[i].dims[1] == ls[i+1].dims[2] || throw(DomainError("Mismatch Dims", "DNN (Inner Constructor): DLayer incompatibility between layers $i and $(i+1)."))
        end
        return new{T}(ls)
    end
end

# Outer Constructor
DNN(ls::Vector{DLayer{T}}) where {T<:Number} = DNN{T}(ls)


"""
	(dnn::DNN{T})(x::AbstractVector{T}) where {T <: Number}
Let `N = DNN.ls[1].dims[2]` and `M = DNN.ls[end].dims[1]`, then
here we treat the structure `DNN` as a function: ``{\\cal R}^N \\mapsto {\\cal R}^M``
Takes input `x` and passes it through each of the layers of `DNN`.

# Type Constraints
- `T <: Number`

# Arguments
- `x :: T`  -- An input value of dimension `N`.

# Return
`::Vector{AD{T}}` of dimension `M`.
"""
function (dnn::DNN{T})(x::AbstractVector{T}) where {T<:Number}

    _, n = size(dnn.layers[1].M)
	length(x) == n || throw(DomainError(n, "DNN (as function): Matrix from first layer is incompatible with `x`."))

    for i in eachindex(dnn.layers)
        x = dnn.layers[i].op.(dnn.layers[i].M * x .+ dnn.layers[i].b)
    end

    return x
end


function make_const!(l::DLayer{T}) where {T<:Number}
    t0 = zero(T)
    n, m = l.dims
    @inbounds for i in 1:n
        l.b[i].d = t0
    end

    @inbounds for i in 1:n
        for j in 1:m
            l.M[i, j].d = t0
        end
    end

    return nothing
end

function set_bd_pd!(l::DLayer{T}, k::Int, d::T) where {T<:Number}
    l.b[k].d = d

    return nothing
end


function set_md_pd!(l::DLayer{T}, k::Int, d::T) where {T<:Number}
    l.M[k].d = d

    return nothing
end


"""
    loss(dnn, X, Y)

Computes the loss of the neural network given inputs, `X`, and outputs `Y`.

# Type Constraints
- `T <: Number`

# Arguments
- `dnn :: DNN{T}`     -- A DNN layer.
- `X   :: Matrix{T}`  -- The matrix of input values.
- `Y   :: Matrix{T}`  -- The matrix of output values.

# Return
`::AD{T}` -- The loss of the network
"""
function loss(dnn::DNN{T}, X::Matrix{T}, Y::Matrix{T}) where {T<:Number}
    _, m = size(X)
    _, my = size(Y)
	m == my || throw(DomainError("Mismatch Dims", "`loss`: Dimensions of `X` and `Y` are incompatible."))

    s = zero(AD{T})
    @inbounds for i in 1:m
        df = dnn(@view X[:, i]) .- (@view Y[:, i])
        s += LA.dot(df, df)
    end

    return s
end


"""
    fit(dnn, X, Y)

Adjusts the parameters of neural network, `dnn`, to get the best fit of 
the data: `X`, `Y`. The parameters of the network are all paris of 
matrices and biases for each layer in the network.

# Type Constraints
- `T <: Number`

# Arguments
- `dnn :: DNN{T}`     -- A DNN layer.
- `X   :: Matrix{T}`  -- The matrix of input values.
- `Y   :: Matrix{T}`  -- The matrix of output values.

# Return
`::nothing`
"""
function fit(dnn::DNN{T}, X::Matrix{T}, Y::Matrix{T};
		N=1000::Int64, relerr=T(1.0e-6)::T, μ=T(1.0e-3)::T, verbose=false::Bool) where {T<:Number}

    _, m = size(X)
    _, my = size(Y)

	m == my || throw(DomainError("Mismatch Dims", "`fit`: Arrays, `X`, and `Y`, are incompatible."))

	lss_last::T = typemax(T)
	lss::T = typemax(T)
	rel_chg::T = typemax(T)
    finished_early = false
    num_iterates::Int64 = N
    mu::T = μ
    @inbounds for j in 1:N
        rel_chg = abs((lss - lss_last) / lss_last)
        if j > 20 && rel_chg <= relerr && lss <= lss_last
			println("rel_chg = $rel_chg")
			println("number of iterates = $j")
            finished_early = true
            num_iterates = j
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
			nn, mm = size(dnn.layers[i].M)
            for k in eachindex(dnn.layers[i].M)

               	set_md_pd!(dnn.layers[i], k, one(T))
               	ls = loss(dnn, X, Y)
               	set_md_pd!(dnn.layers[i], k, zero(T))
               	dnn.layers[i].M[k].v -= ls.d * mu
            end

            # Selectively treat the kth element of b as a variable so that
            # we may take the partial derivative with respect to b[k].
			nn = length(dnn.layers[i].b)
            for k in eachindex(dnn.layers[i].b)
               	set_bd_pd!(dnn.layers[i], k, one(T))
               	ls = loss(dnn, X, Y)
               	set_bd_pd!(dnn.layers[i], k, zero(T))
                lss = ls.v
                dnn.layers[i].b[k].v -= ls.d * brat * mu
            end
        end
    end
    if finished_early
        println("Total number of iterates tried = $num_iterates from a max of $N.")
        println("The relchg = $rel_chg.")
    else
        println("Used the maximum nunmber of iterates ($N).")
        println("The relchg = $rel_chg.")
    end
end

end # DNNS module


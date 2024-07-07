module UtilFunc

include("AutoDiff.jl")
import .AutoDiff: AD

export AD, sigmoid1, sigmoid2, sigmoid3, relu, relur, L1, softmax

#-------------------------------------------------------------------
# ---------  Non Standard and Threshold Functions  -- --------------
#-------------------------------------------------------------------

"""
	sigmoid1(x::AD{T})

Implements an `AD` version of the standard "exponential" sigmoid function.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function sigmoid1(x::AD{T}) where {T<:Number}
    on = one(T)
    v = on / (on + exp(-x.v))
    d = x.d * v * (on - v)

    return AD(v, d)
end


"""
	sigmoid2(x::AD{T})

Implements an `AD` version of the standard "tanh" sigmoid function.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function sigmoid2(x::AD{T}) where {T<:Number}
    v = tanh(x.v)

    return AD(v, x.d * (one(T) - v * v))
end


"""
	sigmoid3(x::AD{T})

Implements an `AD` version of the standard "arctan" sigmoid function.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function sigmoid3(x::AD{T}) where {T<:Number}
    t1 = one(T)
    v = x.v

    return AD(atan(v), x.d * (t1 / (t1 + v * v)))
end


"""
	relu(x::AD{T})

Implements an `AD` version of the standard relu function.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function relu(x::AD{T}) where {T<:Number}
    d = x.v <= 0 ? zero(T) : one(T)
    AD(x.v, d * x.d)
end


"""
	relur(x::AD{T})

Implements an `AD` version of a modified version of the relu function.
The modification is that while the value of the `relur` is the same as `relu`,
its derivative is not. The value of the derivative is `0` or `1`, however
the boundary moves randomly around the natural input boundary of `0`,
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function relur(x::AD{T}) where {T<:Number}
    d = x.v <= rand([-0.25, -0.1, -0.025, -0.01, 0.0, 0.01, 0.025, 0.1, 0.25]) ? zero(T) : one(T)
    AD(x.v, d * x.d)
end



function L1(v::Vector{T}) where {T<:Number}
	n = length(v)
	s = zero(T)
	for i in 1:n
		s += abs(v[i])
	end

	return s
end

function L1(v::Vector{AD{T}}) where {T<:Number}
	n = length(v)
	s = zero(AD{T})
	for i in 1:n
		s += abs(v[i])
	end

	return s
end


"""
	softmax(x::Vector{T} [, τ=one(T)])

Implements the `softmax` function.

# Type Constraints
- T <: Number

# Arguments
- x :: Vector{T}  -- The `AD` input vector.
- τ :: T          -- The "temperature" parameter. 

# Return
::Vector{T} -- The output AD vector.
"""
function softmax(xs::Vector{T}, τ=one(T)::T) where {T<:Number}
	n = length(xs)
	im = argmax([x for x in xs])
	zs = (xs .- xs[im]) / τ
	zsm = zero(T)
	for i in 1:n
		zsm += exp(zs[i])
	end
	return exp.(zs) ./ zsm
end

"""
	softmax(x::Vector{AD{T}} [, τ=one(T)])

Implements an `AD` version of the `softmax` function.

# Type Constraints
- T <: Number

# Arguments
- x :: Vector{AD{T}}  -- The `AD` input vector.
- τ :: T              -- The "temperature" parameter. 

# Return
::Vector{AD{T}} -- The output AD vector.
"""
function softmax(xs::Vector{AD{T}}, τ=one(T)::T) where {T<:Number}
	n = length(xs)
	im = argmax([x.v for x in xs])
	zs = (xs .- xs[im]) / τ
	zsm = AD(zero(T), zero(T))
	for i in 1:n
		zsm += exp(zs[i])
	end
	return exp.(zs) ./ zsm
end


end # module UtilFunc



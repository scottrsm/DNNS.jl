module UtilFunc

import ..AutoDiff: AD
import Random

export sigmoid1, sigmoid2, sigmoid3, relu, relur, L1, softmax

#=----------------------------------------------------------------
---------  Non Standard and Threshold Functions  -- --------------
------------------------------------------------------------------
=#

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


# The random offsets of the derivative boundary used by `relur`.
const RELUR_OFFSETS = (-0.25, -0.1, -0.025, -0.01, 0.0, 0.01, 0.025, 0.1, 0.25)

"""
	relur(x::AD{T}; rng=Random.default_rng())

Implements an `AD` version of a modified version of the relu function.
The modification is that while the value of the `relur` is the same as `relu`,
its derivative is not. The value of the derivative is `0` or `1`, however
the boundary moves randomly around the natural input boundary of `0`
(one of the offsets `±0.25, ±0.1, ±0.025, ±0.01, 0` is drawn from `rng`).

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Keyword Arguments
- rng :: Random.AbstractRNG -- The random number generator used for the boundary offset.

# Return
::AD{T} -- The output AD value/derivative.
"""
function relur(x::AD{T}; rng::Random.AbstractRNG=Random.default_rng()) where {T<:Number}
    d = x.v <= T(rand(rng, RELUR_OFFSETS)) ? zero(T) : one(T)
    AD(x.v, d * x.d)
end


"""
	L1(v::AbstractVector{T})

The ``L_1`` norm (sum of absolute values) of the vector `v`; works for
plain numbers and for `AD` values.

# Return
The sum of the absolute values of the elements of `v` (of the element type of `v`).
"""
function L1(v::AbstractVector{T}) where {T<:Number}
	s = zero(T)
	for x in v
		s += abs(x)
	end

	return s
end


"""
	softmax(x::AbstractVector{T} [, τ=1])

Implements the `softmax` function. Works for plain numbers and,
as an `AD` version, for a vector of `AD` values.

# Type Constraints
- T <: Number

# Arguments
- x :: AbstractVector{T}  -- The input vector (plain numbers or `AD` values).
- τ :: Real               -- The "temperature" parameter (`τ > 0`).

# Return
::Vector{T} -- The output vector.
"""
function softmax(xs::AbstractVector{T}, τ::Real=1) where {T<:Number}
	τ > 0 || throw(DomainError(τ, "softmax: the temperature, `τ`, must be positive."))
	isempty(xs) && throw(DomainError(0, "softmax: the input vector must not be empty."))
	im = argmax(xs)
	zs = (xs .- xs[im]) ./ τ
	zsm = zero(T)
	for z in zs
		zsm += exp(z)
	end
	return exp.(zs) ./ zsm
end


end # module UtilFunc

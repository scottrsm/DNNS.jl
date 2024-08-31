module AutoDiff

export AD

import Base: promote_rule, show
import Base: +, -, *, /, ^, exp, log
import Base: sin, cos, tan, csc, sec, cot
import Base: sinh, cosh, tanh, csch, sech, coth
import Base: asin, acos, atan, acsc, asec, acot
import Base.Threads as TH

import LinearAlgebra as LA
import LinearAlgebra: dot

"""
    AD{T}

Automatic differentiation structure. Essentailly, an
implementation of a *dual* number.

Fields
- v :: T -- The value of this structure.
- d :: T -- The derivative at this value.

"""
mutable struct AD{T<:Number} <: Number
    v::T
    d::T

    # Inner Constructors.
    AD{T}(nv::T, nd::T) where {T<:Number} = new{T}(nv, nd)

    AD{T}(nv::T) where {T<:Number} = new{T}(nv, zero(T))
    function AD{T}(ad::AD{S}) where {T<:Number,S<:Number}
        W = promote_type(S, T)
        return new{W}(convert(W, ad.v), convert(W, ad.d))
    end

    function AD{T}(nv::S) where {T<:Number,S<:Number}
        W = promote_type(S, T)
        return AD{W}(convert(W, nv), zero(W))
    end
end

# Outer Constructors.
AD(nv::T; var::Bool=false) where {T<:Number} = var ? AD{T}(nv, one(T)) : AD{T}(nv, zero(T))
AD(nv::T, nd::T) where {T<:Number} = AD{T}(nv, nd)
AD(nv::T, nd::S) where {T<:Number,S<:Number} = AD(Base.promote(nv, nd)...)

# Show values of AD.
Base.show(io::IO, x::AD{T}) where {T<:Number} = print(io, "($(x.v), $(x.d))")


Base.promote_rule(::Type{AD{T}}, ::Type{T}) where {T<:Number} = AD{T}
Base.promote_rule(::Type{AD{T}}, ::Type{S}) where {T<:Number,S<:Number} = AD{Base.promote_type(T, S)}
Base.promote_rule(::Type{AD{T}}, ::Type{AD{S}}) where {T<:Number,S<:Number} = AD{Base.promote_type(T, S)}


#=------------------------------------------------------------------
------------  Overload Math Functions for AD  ----------------------
--------------------------------------------------------------------
 Binary operators below are defined on two potentially different
 AD types: AD{T}, AD{S}.
 Note: Given the promote_type rules above, we can then do:
 =#



#= -----------------------------------------------------------------
------------  Overload Math Functions for AD  ----------------------
--------------------------------------------------------------------
 Binary operators below are defined on two potentially different
 AD types: AD{T}, AD{S}.
 Note: Given the promote_type rules above, we can then do:
       (operator)(x::AD{T}, y::S})
 The y variable can be promoted to AD{S}, then we have a method match.
 The method (+)(x::AD{T}, AD(y)::AD{S}) gets called.
 The first thing this function does is promote x,y to a promoted_type, W
 (which is invisible in the code) and then a value of type AD{W} is returned.
--------------------------------------------------------------------
=#


#=---------------------------------------------------------------
 -----    Standard Scalar Functions/Operators      --------------
-----------------------------------------------------------------
=#

#= -----  Standard Binary Functions  --------------------------
     Operators: +. -, *, /, ^
=#
function Base.:(+)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    AD(xp.v + yp.v, xp.d + yp.d)
end

function Base.:(-)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    AD(xp.v - yp.v, xp.d - yp.d)
end

function Base.:(*)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    AD(xp.v * yp.v, xp.v * yp.d + xp.d * yp.v)
end

function Base.:(/)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
	yp == zero(eltype(yp.v)) && throw(DomainError(y.v, "AD: 'y.v' == 0 is not a valid value for x.v / y.v."))
    AD(xp.v / yp.v, xp.d / yp.v - (xp.v * yp.d) / (yp.v * yp.v))
end

function Base.:(^)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
	x.v == zero(eltype(xp)) && throw(DomainError(x.v, "AD: 'x.v' == 0 is not a valid value for x.v^y.v."))
    t = xp.v^yp.v
    AD(t, t * (yp.d * log(xp.v) + (yp.v * xp.d) / xp.v))
end


# ----- Standard Unary Scalar Functions  --------------------------
# Unary minus
function Base.:(-)(x::AD{T}) where {T<:Number}
    AD(-x.v, -x.d)
end

# abs
function Base.abs(x::AD{T}) where {T<:Number}
	AD(abs(x.v), x.v >= 0 ? x.d : -x.d)
end

# exp, log
function Base.exp(x::AD{T}) where {T<:Number}
    et = exp(x.v)
    AD(et, et * x.d)
end

function Base.log(x::AD{T}) where {T<:Number}
	x.v == zero(T) && throw(DomainError(x.v, "AD: 'x.v' == 0 is not a valid value for log(x.v)."))
    AD(log(x.v), x.d / x.v)
end



# sin, cos, tan
Base.sin(x::AD{T}) where {T<:Number} = AD(sin(x.v), cos(x.v) * x.d)

Base.cos(x::AD{T}) where {T<:Number} = AD(cos(x.v), -sin(x.v) * x.d)

function Base.tan(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi / 2))
	mx == zero(T) && throw(DomainError(x.v, "AD: 'x.v' mod π / 2 == 0 is not a valid value for tan(x.v)."))
    s = sec(mx)
    t = tan(mx)
    AD(t, s * s * x.d)
end

# csc, sec, cot
function Base.csc(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
	mx == zero(T) && throw(DomainError(x.v, "AD: 'x.v' mod π == 0 is not a valid value for csc(x.v)."))
    ct = cot(mx)
    c = csc(mx)
    AD(c, -c * ct * x.d)
end

function Base.sec(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi / 2))
	mx == zero(T) && throw(DomainError(x.v, "AD: 'x.v' mod π / 2 == 0 is not a valid value for sec(x.v)."))
    s = sec(mx)
    t = tan(mx)
    AD(s, s * t * x.d)
end

function Base.cot(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
	mx == zero(T) && throw(DomainError(x.v, "AD: 'x.v' mod π == 0 is not a valid value for cot(x.v)."))
    c = csc(mx)
    ct = cot(mx)
    AD(ct, -c * c * x.d)
end


# sinh, cosh, tanh
function Base.sinh(x::AD{T}) where {T<:Number}
    ch = cosh(x.v)
    sh = sinh(x.v)
    AD(sh, ch * x.d)
end

function Base.cosh(x::AD{T}) where {T<:Number}
    ch = cosh(x.v)
    sh = sinh(x.v)
    AD(ch, sh * x.d)
end

function Base.tanh(x::AD{T}) where {T<:Number}
    sh = sech(x.v)
    th = tanh(x.v)
    AD(th, sh * sh * x.d)
end

# csch, sech, coth
function Base.csch(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
	mx == zero(T) && throw(DomainError(x.v, "AD: 'x.v' mod π == 0 is not a valid value for csch(x.v)."))
    ch = csch(mx)
    ct = coth(mx)
    AD(ch, -ch * ct * x.d)
end

function Base.sech(x::AD{T}) where {T<:Number}
    sh = sech(x.v)
    th = tanh(x.v)
    AD(sh, -sh * th * x.d)
end

function Base.coth(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
	mx == zero(T) && throw(DomainError(x.v, "AD: 'x.v' mod π == 0 is not a valid value for coth(x.v)."))
    ch = csch(mx)
    ct = coth(mx)
    AD(ct, -ch * ch * x.d)
end



# Inverse trig functions: asin, acos, atan.
Base.asin(x::AD{T}) where {T<:Number} = AD(asin(x.v), one(T) / sqrt(one(T) - x.v * x.v))
Base.acos(x::AD{T}) where {T<:Number} = AD(acos(x.v), -one(T) / sqrt(one(T) - x.v * x.v))
Base.atan(x::AD{T}) where {T<:Number} = AD(atan(x.v), one(T) / (one(T) + x.v * x.v))

# Inverse trig functions: acsc, asec, acot.
function Base.acsc(x::AD{T}) where {T<:Number}
	abs(x.v) < one(T) && throw(DomainError(x.v, "AD: '|x.v|' < 1 is not a valid value for asec(x.v)."))
    AD(acsc(x.v), -one(T) / (x.v * sqrt(x.v * x.v - one(T))))
end

function Base.asec(x::AD{T}) where {T<:Number}
	abs(x.v) < one(T) && throw(DomainError(x.v, "AD: '|x.v|' < 1 is not a valid value for asec(x.v)."))
    AD(asec(x.v), one(T) / (abs(x.v) * sqrt(x.v * x.v - one(T))))
end

Base.acot(x::AD{T}) where {T<:Number} = AD(acot(x.v), -one(T) / (one(T) + x.v * x.v))



#=-----------------------------------------------------------------
---------       Matrix/Vector Functions               ------------
-------------------------------------------------------------------
=#
# NOTE: We need to essentially duplicate a few functions below.
#       Julia is *NOT* covariant with respect to parametric types like Vector{T}.
#       That is, it is *NOT* true that: Vector{S} <: Vector{T} if T <: S.
#       We don't even have AD{T} <: T, but even if we did we still wouldn't have 
#       Vector{AD{T}} <: Vector{T}.

Base.zero(::Type{AD{T}}) where {T<:Number} = AD(zero(T), zero(T))
Base.zeros(::Type{AD{T}}, n::Int) where {T<:Number} = fill(AD(zero(T), zero(T), n))


function LA.dot(x::Vector{AD{T}}, y::Vector{AD{T}}) where {T<:Number}
    n = length(x)
    if length(y) != n
		throw(DomainError("Incompatible dims", "dot: Vector lengths are not the same."))
    end
    s = zero(AD{T})
    @inbounds @simd for i in 1:n
        s += x[i] * y[i]
    end
    return s
end

function LA.dot(x::Vector{AD{T}}, y::Vector{T}) where {T<:Number}
    n = length(x)
    if length(y) != n
		throw(DomainError("Incompatible dims", "dot: Vector lengths are not the same."))
    end
    s = zero(AD{T})
    @inbounds @simd for i in 1:n
        s += x[i] * y[i]
    end
    return s
end

function LA.dot(x::Vector{T}, y::Vector{AD{T}}) where {T<:Number}
    return LA.dot(y, x)
end



# Extend Matrix/vector multiplication to AD{T}/AD{T}.
function Base.:(*)(A::Matrix{AD{T}}, v::Vector{AD{T}}) where {T<:Number}
    n, m = size(A)
    if m != length(v)
		throw(DomainError("Incompatible dims", "*: Matrix A and vector v have incompatible sizes."))
    end

    res = Vector{AD{T}}(undef, n)

    @inbounds for i in 1:n
        s = zero(AD{T})
        @simd for j in 1:m
            s += A[i, j] * v[j]
        end
        res[i] = s
    end

    return res
end

# Extend Matrix/vector multiplication to AD{T}/T.
function Base.:(*)(A::Matrix{AD{T}}, v::Vector{T}) where {T<:Number}
    n, m = size(A)
    if m != length(v)
		throw(DomainError("Incompatible dims", "*: Matrix A and vector v have incompatible sizes."))
    end

    res = Vector{AD{T}}(undef, n)

    @inbounds for i in 1:n
        s = zero(AD{T})
        @simd for j in 1:m
            s += A[i, j] * v[j]
        end
        res[i] = s
    end

    return res
end


# Extend isapprox to AD{T}.
Base.isapprox(x::AD{T}, y::AD{T}; rtol) where {T <: Number} = (abs(x.d - y.d) <= rtol) && (abs(x.v - y.v) <= rtol)

end # module AutoDiff




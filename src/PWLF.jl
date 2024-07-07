module PWLF

include("AutoDiff.jl")
import .AutoDiff: AD

export PWL

const X_REL_TOL = 1.0e-6

"""
    PWL{T}

A structure representing a piece-wise linear function on the Real line.

In practice, one uses one of two outer constructors to create a `PWL` struct.

## Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.

## Fields
- `xs :: Vector{T}`  -- The "x" values.
- `ys :: Vector{T}`  -- The "y" values.
- `ds :: Vector{T}`  -- The "slopes" of the segments. Including the left most and right most slopes.
- `n  :: Int`        -- The number of "x,y" values.
                         

## Public Constructors
`PWL(xs::Vector{T}, y::T, ds::Vector{T})` 
- `xs` -- The `x` coordinates in ascending order -- no duplicates.
- `y`  -- The value of `y` corresponding to the first entry in `xs`.
- `ds` -- The slopes of all "x" intervals as well as the "left" slope of the first
          point and the "right" slope of the last point.

`PWL(xs::Vector{T}, ys::Vector{T}, ds::Vector{T})`
- `xs` -- The `x` coordinates in ascending order -- no duplicates.
- `ys` -- The `y` coordinates corresponding to each `x` value.
- `ds` -- A 2-Vector consisting of the "left" slope of the first point and the "right"
          slope of the last point.

## Examples
```jdoctest
julia> # Create the same (in behavior) Piecewise linear functions in two ways:
julia> pw1 = PWL([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.0, 5.0])

julia> pw2 = PWL([1.0, 2.0, 3.0], 2.0, [0.0, 1.0, 1.0, 5.0])

julia> pw1(2.5)
3.5

julia> pw2(2.5)
3.5
```
"""
struct PWL{T<:Number}
    xs::Vector{T}
    ys::Vector{T}
    ds::Vector{T}
    n::Int

    # Inner Constructor.
    function PWL{T}(nxs::Vector{T}, nys::Vector{T}, ndx::Vector{T}) where {T<:Number}
		# Check that type, T, has a total ordering.
		isTotalOrder(T) || throw(DomainError(T, "`PWD{T}`: (Inner Constructor) Type `$T` does not have a total ordering."))

        xmin, xmax = extrema(nxs)
        tol = X_REL_TOL * max(abs(xmin), abs(xmax))

        n = length(nxs)
        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError(nxs, "`PWD{T}`: (Inner Constructor) `nxs` is not sorted or has duplicates."))
        end

        if length(ndx) != 2
            throw(DomainError(nls, "`PWD{T}`: (Inner Constructor) `ndx` vector must have a length of 2."))
        end

        n = length(nxs)
        if n != length(nys)
            throw(DomainError(nys, "`PWD{T}`: (Inner Constructor) `nxs` and `nys` vectors must have the same length."))
        end

        # Compute the interior slopes.
        dxs = diff(nys) ./ diff(nxs)

        W = eltype(dxs)
        if W != T
            nxs = convert.(W, nxs)
            nys = convert.(W, nys)
            dxs = convert.(W, dxs)
            ndx = convert.(W, ndx)
        end

        nds = Vector{W}(undef, n + 1)
        nds[1]   = ndx[1]
        nds[n+1] = ndx[2]
        nds[2:n] = dxs

        return new{W}(copy(nxs), copy(nys), nds, n)

    end

    # Inner Constructor.
    function PWL{T}(nxs::Vector{T}, ny::T, nls::Vector{T}) where {T<:Number}
		# Check that type, T, has a total ordering.
		isTotalOrder(T) || throw(DomainError(T, "`PWD{T}`: (Inner Constructor) Type `$T` does not have a total ordering."))

        n = length(nxs)
        xmin, xmax = extrema(nxs)
        tol = X_REL_TOL * max(abs(xmin), abs(xmax))

        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError(nxs, "`PWD{T}`: (Inner Constructor) `nxs` is not sorted or has duplicates."))
        end

        if (length(nls) - n) != 1
            throw(DomainError(nls, "`PWD{T}`: (Inner Constructor) `nls` vector length must be 1 larger than `nxs` vector length."))
        end

        nys = zeros(T, length(nxs))
        lasty = ny
        nys[1] = lasty

        if isbitstype(T)
            for i in 2:n
                lasty += (nxs[i] - nxs[i-1]) * nls[i]
                nys[i] = lasty
            end
        else
            for i in 2:n
                lasty += (nxs[i] - nxs[i-1]) * nls[i]
                nys[i] = deepcopy(lasty)
            end
        end

        new(copy(nxs), nys, copy(nls), n)
    end
end


# Outer Constructors.
PWL(nxs::Vector{T}, ny::T, nls::Vector{T}) where {T<:Number} = PWL{T}(nxs, ny, nls)

# Outer constructor.
PWL(nxs::Vector{T}, nys::Vector{T}, nds::Vector{T}) where {T<:Number} = PWL{T}(nxs, nys, nds)



"""
	(PWL{T})(x::T) where {T<:Number}

Uses the structure `PWL` as a piece-wise linear function. 

# Type Constraints
- `T <: Number`

# Arguments
- `x :: T`  -- An input value.

# Return
`:: T`
"""
function (p::PWL{T})(x::T) where {T<:Number}

    idx = searchsorted(p.xs, x)
    u = first(idx)
    l = last(idx)

    l += l == 0 ? 1 : 0

    return p.ys[l] + p.ds[u] * (x - p.xs[l])
end


"""
	(PWL{T})(x::AD{T}) where {T<:Number}

Uses the structure `PWL` as a piece-wise linear function. 

# Type Constraints
- `T <: Number`

# Arguments
- `x :: AD{T}`  -- An AutoDiff value.

# Return
`:: AD{T}`
"""
function (p::PWL{T})(x::AD{T}) where {T<:Number}

    Idx = searchsorted(p.xs, x.v)
    u = first(Idx)
    l = last(Idx)

    l += l == 0 ? 1 : 0

    return AD(p.ys[l] + p.ds[u] * (x.v - p.xs[l]), p.ds[u] * x.d)
end

isTotalOrder(::Type{Complex})  = false
isTotalOrder(::Type{<:Number}) = true

end # module PWLF


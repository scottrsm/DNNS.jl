module PWLF

include("AutoDiff.jl")
import ..AutoDiff: AD

import Base
import Plots

export PWL, smooth

const X_REL_TOL = 1.0e-6

"""
    PWL{T}

A structure representing a piece-wise linear function on the Real line.

In practice, one uses one of two outer constructors to create a `PWL` struct.

# Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.

# Fields
- `xs :: Vector{T}`  -- The "x" values.
- `ys :: Vector{T}`  -- The "y" values.
- `ds :: Vector{T}`  -- The "slopes" of the segments. Including the left most and right most slopes.
- `n  :: Int`        -- The number of "x,y" values.
                         
# Input Contract
- `xs` -- Must be in strict ascending order.
- `xs` -- `` |\\bf{xs}| \\ge 2``.
- `ys` -- `` |\\bf{xs}| = |\\bf{ys}|``.
- `ds` -- `` |\\bf{ds}| = |\\bf{xs}| + 1``.

# Public Constructors
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

# Examples
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

		# Check Input Contract...
		
		# Check that type, T, has a total ordering.
		isTotalOrder(T) || throw(DomainError(T, "`PWD{T}`: (Inner Constructor) Type `$T` does not have a total ordering."))

        xmin, xmax = extrema(nxs)
        tol = X_REL_TOL * max(abs(xmin), abs(xmax))

		# Check that `nxs` is in strict increasing order.
        n = length(nxs)
        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError(nxs, "`PWD{T}`: (Inner Constructor) `nxs` is not a strictly increasing sequence."))
        end

		# Check that `nxs` has length >= 2.
        if n < 2
            throw(DomainError(nxs, "`PWD{T}`: (Inner Constructor) `nxs` vector must have a length of at least 2."))
        end

		# Check that `nxs` and `nys` have the same length.
        if n != length(nys)
            throw(DomainError(nys, "`PWD{T}`: (Inner Constructor) `nxs` and `nys` vectors must have the same length."))
        end
		
		# Check that the length of `ndx` is 2.
        if length(ndx) != 2
			throw(DomainError(nys, "`PWD{T}`: (Inner Constructor) The length of `ndx` must have length 2."))
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
    function PWL{T}(nxs::Vector{T}, ny::T, nds::Vector{T}) where {T<:Number}

		# Check the Input Contract...
		#
		# Check that type, T, has a total ordering.
		isTotalOrder(T) || throw(DomainError(T, "`PWD{T}`: (Inner Constructor) Type `$T` does not have a total ordering."))

        n = length(nxs)
        xmin, xmax = extrema(nxs)
        tol = X_REL_TOL * max(abs(xmin), abs(xmax))

		# Check that `nxs` has length >= 2.
        if n < 2
            throw(DomainError(nls, "`PWD{T}`: (Inner Constructor) `nxs` vector must have a length of at least 2."))
        end

		# Check that `nxs` is in strict ascending order.
        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError(nxs, "`PWD{T}`: (Inner Constructor) `nxs` is not sorted or has duplicates."))
        end

		# Check that the length of `dns` is 1 more than the length of `nxs'.
        if n + 1 != length(nds)
			throw(DomainError(nds, "`PWD{T}`: (Inner Constructor) The length of  `nds` must be 1 more than the length of `nxs`."))
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

isTotalOrder(::Type{Complex})     = false
isTotalOrder(::Type{AD{Complex}}) = false
isTotalOrder(::Type{<:Number})    = true

"""
	merge(p1::PWL{T}, p2::PWL{T})

Merges two `PWL` objects into one.
The new object contains all of the `x` (node) values.
If the two objects have values for the same `x` (node) the
corresponding `y` value will be the average of the two `y` values.

# Type Constraints
- `T <: Number`

# Arguments
- `p1::PWL{T} -- First `PWL` object.
- `p2::PWL{T} -- Second `PWL` object.

# Return
- ::PWL{T} -- A combined `PWL` object.
"""
function Base.merge(p1::PWL{T}, p2::PWL{T}) :: PWL{T} where {T <: Number}
    d1 = OrderedDict(zip(p1.xs, p1.ys))
    d2 = OrderedDict(zip(p2.xs, p2.ys))
    d = sort(merge(d1, d2))
    for k in keys(d)
		d[k] = (d[k] + d[k]) / T(2)
    end
    p1_min = p1.xs[1]
    p1_max = p1.xs[end]
    p2_min = p2.xs[1]
    p2_max = p2.xs[end]
    ds1 = p1.xs[1  ] < p2.xs[1  ] ? p1.ds[1  ] : p2.ds[1  ]
    ds2 = p1.xs[end] < p2.xs[end] ? p2.ds[end] : p1.ds[end]

    return PWL(collect(keys(d)), collect(values(d)), [ds1, ds2])
end


"""
	smooth(p::PWL{T}, Δ::T)

Potentially smooths the `PWL` object by combining adjacent `x` nodes
if the distance between them is less than Δ.
The new `x` node replaces the other two and is half way between the other nodes.
The new `y` value is the average of the `y` values of the other two nodes.

# Type Constraints
- `T <: Number`

# Arguments
- `p::PWL{T}` -- PWL object.
- `Δ::T`      -- The smoothing window.

# Returns
- `::PWL{T}` -- New, potentially smoothed `PWL` object.
"""
function smooth(p, Δ)
    xl = p.xs[1]
    n = length(p.xs)
    is = Int[]
    for i in 2:n
        if p.xs[i] - xl < Δ
            push!(is, i)
        end
    end
    cnt = 0
    li = 0
    for i in 1:length(is)
        if i == 1 && li != 1
            cnt += 1
        end
        li = i
    end
    xs = Vector{Float64}(undef, n - cnt)
    ys = Vector{Float64}(undef, n - cnt)
    j = 1
    i = 1
    while i <= n
        if i in is
            xs[j] = (p.xs[i] + p.xs[i-1]) / 2.0
            ys[j] = (p.ys[i] + p.ys[i-1]) / 2.0
            i += 1
        else
            xs[j] = p.xs[i]
            ys[j] = p.ys[i]
        end
        j += 1
        i += 1
    end

    return PWL(xs, ys, [p.ds[1], p.ds[end]])
end

"""
	plot(p::PWL{T}; <keywords>)

Plots a `PWL` object.

# Arguments
- `p::PWL{T}` -- A `PWL` object 

# Keyword Arguments
- `label::Union{String, Nothing}` -- Label of the graph
- `lw::Int`    -- Line width.
- `lc::Symbol` -- Line color.
- `ec::Symbol  -- Edge color.
- `es::Symbol  -- Edge drawing style.

"""
function Plots.plot(p::PWL{T}; label=nothing, lc=:blue, ec=:red, lw=1, es=:dash) where {T <: Number}
    Plots.plot(p.xs, p.ys, label=label, lw=2)
    dt = convert(T, 0.1) * (p.xs[end] - p.xs[1])
    Plots.plot!([p.xs[1] - dt, p.xs[1]], [p.ys[1], p.ys[1] - p.ds[1] * dt], lc=ec, ls=es, label=nothing)
    Plots.plot!([p.xs[end], p.xs[end] + dt], [p.ys[end], p.ys[end] + p.ds[end] * dt], lc=ec, ls=es, label=nothing)
end


end # module PWLF


module PWLF

import ..AutoDiff: AD

import Base: merge
import Plots
import OrderedCollections: OrderedDict

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
Inputs are promoted to a common floating point type (integer inputs give a `PWL{Float64}`).

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
    function PWL{T}(nxs::AbstractVector{T}, nys::AbstractVector{T}, ndx::AbstractVector{T}) where {T<:Number}

		# Check Input Contract...
		
		# Check that type, T, has a total ordering.
		isTotalOrder(T) || throw(DomainError(T, "`PWL{T}`: (Inner Constructor) Type `$T` does not have a total ordering."))

		# Check that `nxs` has length >= 2.
        n = length(nxs)
        if n < 2
            throw(DomainError(nxs, "`PWL{T}`: (Inner Constructor) `nxs` vector must have a length of at least 2."))
        end

		# Check that `nxs` and `nys` have the same length.
        if n != length(nys)
            throw(DomainError(nys, "`PWL{T}`: (Inner Constructor) `nxs` and `nys` vectors must have the same length."))
        end

		# Check that `ndx` holds the two end slopes.
        if length(ndx) != 2
            throw(DomainError(ndx, "`PWL{T}`: (Inner Constructor) `nds` must hold exactly the two end slopes."))
        end

		# Check that `nxs` is in strict increasing order.
        tol = _x_tol(nxs)
        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError(nxs, "`PWL{T}`: (Inner Constructor) `nxs` is not a strictly increasing sequence."))
        end

        # Compute the interior slopes.
        dxs = diff(nys) ./ diff(nxs)

        nds = Vector{T}(undef, n + 1)
        nds[1]   = ndx[1]
        nds[n+1] = ndx[2]
        nds[2:n] = dxs

        return new{T}(collect(T, nxs), collect(T, nys), nds, n)

    end

    # Inner Constructor.
    function PWL{T}(nxs::AbstractVector{T}, ny::T, nds::AbstractVector{T}) where {T<:Number}

		# Check the Input Contract...
		#
		# Check that type, T, has a total ordering.
		isTotalOrder(T) || throw(DomainError(T, "`PWL{T}`: (Inner Constructor) Type `$T` does not have a total ordering."))

		# Check that `nxs` has length >= 2.
        n = length(nxs)
        if n < 2
            throw(DomainError(nxs, "`PWL{T}`: (Inner Constructor) `nxs` vector must have a length of at least 2."))
        end

		# Check that `nxs` is in strict ascending order.
        tol = _x_tol(nxs)
        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError(nxs, "`PWL{T}`: (Inner Constructor) `nxs` is not sorted or has duplicates."))
        end

		# Check that the length of `nds` is 1 more than the length of `nxs'.
        if n + 1 != length(nds)
			throw(DomainError(nds, "`PWL{T}`: (Inner Constructor) The length of  `nds` must be 1 more than the length of `nxs`."))
        end

        nys = zeros(T, n)
        lasty = ny
        nys[1] = lasty

        for i in 2:n
            lasty += (nxs[i] - nxs[i-1]) * nds[i]
            nys[i] = lasty
        end

        new{T}(collect(T, nxs), nys, collect(T, nds), n)
    end
end

# The absolute tolerance used to decide whether two `x` nodes coincide.
function _x_tol(xs::AbstractVector{T}) where {T<:Number}
    xmin, xmax = extrema(xs)
    return X_REL_TOL * max(abs(xmin), abs(xmax))
end

# The common (floating point) element type of the inputs.
_pwl_type(types...) = (W = promote_type(types...); typeof(one(W) / one(W)))

# Outer Constructors: promote all inputs to a common floating point type.
function PWL(nxs::AbstractVector{<:Number}, ny::Number, nls::AbstractVector{<:Number})
    W = _pwl_type(eltype(nxs), typeof(ny), eltype(nls))
    return PWL{W}(collect(W, nxs), W(ny), collect(W, nls))
end

function PWL(nxs::AbstractVector{<:Number}, nys::AbstractVector{<:Number}, nds::AbstractVector{<:Number})
    W = _pwl_type(eltype(nxs), eltype(nys), eltype(nds))
    return PWL{W}(collect(W, nxs), collect(W, nys), collect(W, nds))
end



"""
	(PWL{T})(x::Real) where {T<:Number}

Uses the structure `PWL` as a piece-wise linear function. 

# Type Constraints
- `T <: Number`

# Arguments
- `x :: Real`  -- An input value (converted to `T`).

# Return
`:: T`
"""
(p::PWL{T})(x::Real) where {T<:Number} = _pwl_eval(p, convert(T, x))

function _pwl_eval(p::PWL{T}, x::T) where {T<:Number}

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
function (p::PWL{T})(x::AD) where {T<:Number}
    x = AD{T}(x)

    Idx = searchsorted(p.xs, x.v)
    u = first(Idx)
    l = last(Idx)

    l += l == 0 ? 1 : 0

    return AD(p.ys[l] + p.ds[u] * (x.v - p.xs[l]), p.ds[u] * x.d)
end

isTotalOrder(::Type{AD{T}}) where {T<:Number} = isTotalOrder(T)
isTotalOrder(::Type{<:Real})       = true
isTotalOrder(::Type{<:Number})     = false

"""
	merge(p1::PWL{T}, p2::PWL{T})

Merges two `PWL` objects into one.
The new object contains all of the `x` (node) values.
If the two objects have values for the same `x` (node) -- nodes closer than
the relative tolerance used by the `PWL` constructor count as the same -- the
corresponding `x`, `y` values will be the values from `p2`.
If either end node of `p1` and `p2` coincide, take, for that node,
the derivative value from the corresponding node of `p2`.

# Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.

# Arguments
- `p1::PWL{T} -- First `PWL` object.
- `p2::PWL{T} -- Second `PWL` object.

# Return
- ::PWL{T} -- A combined `PWL` object.
"""
function Base.merge(p1::PWL{T}, p2::PWL{T}) :: PWL{T} where {T <: Number}
	# Check that type, T, has a total ordering.
	isTotalOrder(T) || throw(DomainError(T, "`merge`: Type `$T` does not have a total ordering."))

	# Merge the two vectors of `(x,y)` pairs from the two piece-wise linear functions. 
	# In case of duplicate `x` nodes (within tolerance), `p2`'s node will be chosen.
	tol = _x_tol(vcat(p1.xs, p2.xs))
	d = OrderedDict{T, T}()
	for (x, y) in zip(p1.xs, p1.ys)
		any(x2 -> abs(x2 - x) <= tol, p2.xs) || (d[x] = y)
	end
	for (x, y) in zip(p2.xs, p2.ys)
		d[x] = y
	end
    sort!(d)

	# Establish whose end points to use.
    p1_min = p1.xs[1]
    p1_max = p1.xs[end]
    p2_min = p2.xs[1]
    p2_max = p2.xs[end]

	# If the smallest `x` node from `p1` is *strictly* less than the smallest `x` node from `p2`,
	# 	pick `p1`'s first derivative; otherwise use `p2`'s first derivative.
    ds1 = p1.xs[1  ] < p2.xs[1  ] ? p1.ds[1  ] : p2.ds[1  ]

	# If the largest `x` node from `p1` is *strictly* greater than the largest `x` node from `p2`,
	# 	pick `p1`'s last derivative; otherwise use `p2`'s last derivative.
    ds2 = p1.xs[end] > p2.xs[end] ? p1.ds[end] : p2.ds[end]

	# If the first `x` node of `p1` and `p2` coincide, take the first derivative value from `p2`.
	ds1 = abs(p1.xs[1  ] - p2.xs[1  ]) <= tol ? p2.ds[1  ] : ds1

	# If the last `x` node of `p1` and `p2` coincide, take the last derivative value from `p2`.
    ds2 = abs(p1.xs[end] - p2.xs[end]) <= tol ? p2.ds[end] : ds2

	# Use the constructor with the `x`, `y` values along with the
	# end point derivatives.
    return PWL(collect(keys(d)), collect(values(d)), [ds1, ds2])
end


"""
	smooth(p::PWL{T}, Δ::T)

Potentially smooths the `PWL` object by combining adjacent `x` nodes
if the distance between them is less than Δ.
A run of consecutive nodes, each closer than Δ to the previous one, is replaced by
a single node whose `x` and `y` values are the averages of the `x` and `y`
values of the nodes in the run. (For two close nodes the new node is half way between them.)
The function returns a new (potentially) smoothed `PWL` struct.

# Type Constraints
- `T <: Number`
- The type `T` must have a total ordering.

# Arguments
- `p::PWL{T}` -- PWL object.
- `Δ::T`      -- The smoothing window.

# Returns
- `::PWL{T}` -- New, potentially smoothed `PWL` struct.
"""
function smooth(p::PWL{T}, Δ::T) where {T <: Number}
	# Check that type, T, has a total ordering.
	isTotalOrder(T) || throw(DomainError(T, "`smooth`: Type `$T` does not have a total ordering."))

    xl = p.xs[1]
    n = p.n

	# If none of the points are close enough to warrant smoothing, return a copy of `p`.
	if !any(i -> p.xs[i] - p.xs[i-1] < Δ, 2:n)
		return deepcopy(p)
	end

	# Build the new `x`, `y` vectors: walk the nodes, gathering runs of close nodes
	# and replacing each run by the average of its nodes.
    xs = T[]
    ys = T[]
    i = 1
    while i <= n
        # Find the end, `k`, of the run of close nodes starting at `i`.
        k = i
        while k < n && p.xs[k+1] - p.xs[k] < Δ
            k += 1
        end
        cnt = T(k - i + 1)
        push!(xs, sum(@view p.xs[i:k]) / cnt)
        push!(ys, sum(@view p.ys[i:k]) / cnt)
        i = k + 1
    end

	# Return the new (potentially) smoothed PWL.
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
    Plots.plot(p.xs, p.ys, label=label, lw=lw, lc=lc)
    dt = convert(T, 0.1) * (p.xs[end] - p.xs[1])
    Plots.plot!([p.xs[1] - dt, p.xs[1]], [p.ys[1], p.ys[1] - p.ds[1] * dt], lc=ec, ls=es, label=nothing)
    Plots.plot!([p.xs[end], p.xs[end] + dt], [p.ys[end], p.ys[end] + p.ds[end] * dt], lc=ec, ls=es, label=nothing)
end

# Extend isapprox to PWL{T}: same number of nodes, and approximately equal nodes, values and slopes.
function Base.isapprox(p1::PWL, p2::PWL; kwargs...)
	(p1.n == p2.n) && 
	isapprox(p1.xs, p2.xs; kwargs...) && 
	isapprox(p1.ys, p2.ys; kwargs...) && 
	isapprox(p1.ds, p2.ds; kwargs...) 
end

# Structural equality.
Base.:(==)(p1::PWL, p2::PWL) = p1.n == p2.n && p1.xs == p2.xs && p1.ys == p2.ys && p1.ds == p2.ds

end # module PWLF


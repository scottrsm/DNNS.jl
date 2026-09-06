using Test
using Random

using DNNS

import DNNS: AutoDiff, PWLF, UtilFunc
import DNNS.AutoDiff: AD
import DNNS.PWLF: PWL, smooth

const TOL = 1.0e-10

# Fildelity checks for all modules.
@testset "DNNS (Fidelity)" begin
    @test length(detect_ambiguities(DNNS)) == 0
end

@testset "AutoDiff (Fidelity)" begin
    @test length(detect_ambiguities(AutoDiff)) == 0
end

@testset "PWLF (Fidelity)" begin
    @test length(detect_ambiguities(PWLF)) == 0
end

@testset "UtilFunc (Fidelity)" begin
    @test length(detect_ambiguities(UtilFunc)) == 0
end


# Check AutoDiff module
@testset "AutoDiff (AutoDiff Calculations)" begin
	x = AD{Float64}(1.0, 1.0)
	@test sin(x)  ≈ AD{Float64}(0.8414709848078965, 0.5403023058681398) rtol=TOL

	x = AD{Float64}(π / 4.0, 2.0)
	@test tan(x)  ≈ AD{Float64}(1.0, 4.0) rtol=TOL 

	poly(x::AD{Float64}) = 1.0 + 3.0 * x^2 + 2.0 * x^3

	y = poly(AD{Float64}(2.0, 1.0)) 
	@test y ≈  AD{Float64}(29.0, 36.0) rtol=TOL

	x = AD{Float64}(1.0, 2.0)
	y = AD{Float64}(3., 1.0)
	@test x^y ≈  AD{Float64}(1.0, 6.0) rtol=TOL

	# Arithmetic with plain numbers, and the remaining elementary functions.
	x = AD(2.0, 1.0)
	@test x + 1 == AD(3.0, 1.0) && 1 + x == AD(3.0, 1.0)
	@test x - 1 == AD(1.0, 1.0) && 1 - x == AD(-1.0, -1.0)
	@test 3 * x == AD(6.0, 3.0) && -x == AD(-2.0, -1.0)
	@test x / 2 == AD(1.0, 0.5)
	@test 1 / x ≈ AD(0.5, -0.25)
	@test exp(x) ≈ AD(exp(2.0), exp(2.0))
	@test log(x) ≈ AD(log(2.0), 0.5)
	@test sqrt(AD(4.0, 1.0)) ≈ AD(2.0, 0.25)
	@test abs(AD(-2.0, 1.0)) == AD(2.0, -1.0)
	@test sinh(x) ≈ AD(sinh(2.0), cosh(2.0)) && cosh(x) ≈ AD(cosh(2.0), sinh(2.0))
	@test tanh(x) ≈ AD(tanh(2.0), sech(2.0)^2)
	@test AD(1, 2) isa AD{Int} && AD(1, 2.5) isa AD{Float64}
	@test AD(3.0; var=true) == AD(3.0, 1.0) && AD(3.0) == AD(3.0, 0.0)
end

@testset "AutoDiff (Equality, conversion, guards)" begin
	# Structural equality and hashing.
	@test AD(1.0, 0.0) == AD(1.0, 0.0)
	@test AD(1.0, 0.0) != AD(1.0, 1.0)
	@test isequal(AD(1.0, 0.0), AD(1.0, 0.0))
	@test hash(AD(1.0, 0.0)) == hash(AD(1.0, 0.0))
	@test length(Set([AD(1.0, 0.0), AD(1.0, 0.0)])) == 1
	@test AD(1.0, 1.0) ≈ AD(1.0, 1.0)
	@test AD(1.0, 1.0) ≈ AD(1.0 + 1e-13, 1.0)
	@test !(AD(1.0, 1.0) ≈ AD(1.0, 1.1))
	@test isapprox(AD(1.0, 1.0), AD(1.0, 1.0 + 1e-9); atol=1e-8)
	@test AD(1.0, 0.0) == 1.0 && zero(AD{Float64}) == 0 && one(AD{Float64}) == 1

	# The type parameter is authoritative in `AD{T}` constructors and conversions.
	@test AD{Float32}(1.0) isa AD{Float32}
	@test AD{Float32}(AD(1.0, 1.0)) isa AD{Float32}
	@test convert(AD{Float32}, AD(1.0, 1.0)) === AD{Float32}(1.0f0, 1.0f0)
	v = Vector{AD{Float32}}(undef, 1); v[1] = AD(1.0, 1.0)
	@test v[1] isa AD{Float32}

	# `zeros` gives independent values.
	z = zeros(AD{Float64}, 3)
	@test z == [AD(0.0, 0.0), AD(0.0, 0.0), AD(0.0, 0.0)]
	z[1] = AD(5.0, 0.0)
	@test z[2] == AD(0.0, 0.0)

	# Division by zero and powers.
	@test_throws DomainError AD(1.0, 1.0) / 0.0
	@test_throws DomainError AD(1.0, 1.0) / AD(0.0, 1.0)
	@test AD(0.0, 1.0)^4.0 == AD(0.0, 0.0)
	@test AD(-2.0, 1.0)^4.0 == AD(16.0, -32.0)
	@test AD(2.0, 1.0)^3 == AD(8.0, 12.0)
	@test AD(2.0, 1.0)^0.0 == AD(1.0, 0.0)
	@test AD(2.0, 1.0)^AD(3.0, 1.0) ≈ AD(8.0, 8.0 * (log(2.0) + 1.5))
	@test_throws DomainError AD(-2.0, 1.0)^AD(3.0, 1.0)
	@test_throws DomainError log(AD(0.0, 1.0))
	@test_throws DomainError sqrt(AD(0.0, 1.0))
	@test_throws DomainError acsc(AD(0.5, 1.0))
end

# Check PWLF module
@testset "PWLF (PWLF Calculations)" begin
	p1 = PWL([1.0, 3.0, 4.0, 6.0], [2.0, 4.0, 6.0, 10.0], [0.0, 1.0])
	p2 = PWL([1.0, 3.0, 4.0, 6.0], [3.0, 30.0, 3.0, 10.0], [-2.0, 5.0])
	pm = merge(p1, p2)
	ps = PWLF.smooth(pm, 1.5)

	@test pm ≈ PWL([1.0, 3.0, 4.0, 6.0], [3.0, 30.0, 3.0, 10.0], [-2.0, 5.0])  rtol=TOL 
	@test ps ≈ PWL([1.0, 3.5, 6.0], [3.0, 16.5, 10.0], [-2.0, 5.0])            rtol=TOL
	@test ps == PWL([1.0, 3.5, 6.0], [3.0, 16.5, 10.0], [-2.0, 5.0])

	# Evaluation: at nodes, between nodes, and beyond both ends (using the end slopes).
	pw1 = PWL([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.0, 5.0])
	pw2 = PWL([1.0, 2.0, 3.0], 2.0, [0.0, 1.0, 1.0, 5.0])
	@test pw1 ≈ pw2
	@test pw1(2.5) == 3.5 && pw2(2.5) == 3.5
	@test pw1(2.0) == 3.0 && pw1(1.0) == 2.0 && pw1(3.0) == 4.0
	@test pw1(0.0) == 2.0            # left slope 0
	@test pw1(4.0) == 9.0            # right slope 5
	@test pw1(2) == 3.0              # integer input
	@test pw1(AD(2.5, 1.0)) == AD(3.5, 1.0)
	@test pw1(AD(4.0, 2.0)) == AD(9.0, 10.0)

	# Integer inputs give a floating point PWL; the constructors return the stated type.
	pi = PWL([1, 2, 3], [1, 2, 3], [0, 1])
	@test pi isa PWL{Float64} && pi(1.5) == 1.5
	@test PWL{Float64}([1.0, 2.0], [1.0, 2.0], [0.0, 0.0]) isa PWL{Float64}
	@test PWL(Float32[1, 2], Float32[1, 2], Float32[0, 0]) isa PWL{Float32}

	# Smoothing: several separated clusters, and a run of three close nodes (averaged into one).
	p = PWL([1.0, 1.1, 2.0, 2.1, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0])
	ps2 = smooth(p, 0.5)
	@test ps2.xs ≈ [1.05, 2.05, 5.0] && ps2.ys ≈ [1.5, 3.5, 5.0]
	p3 = smooth(PWL([1.0, 1.1, 1.2, 3.0], [1.0, 2.0, 3.0, 4.0], [0.0, 1.0]), 0.5)
	@test p3.xs ≈ [1.1, 3.0] && p3.ys ≈ [2.0, 4.0]
	@test smooth(PWL([1.0, 3.0], [1.0, 2.0], [0.0, 0.0]), 0.5) == PWL([1.0, 3.0], [1.0, 2.0], [0.0, 0.0])

	# Merging: disjoint ranges, coincident end points, and nodes equal within tolerance.
	q1 = PWL([1.0, 2.0], [1.0, 2.0], [0.0, 0.0])
	q2 = PWL([3.0, 4.0], [5.0, 6.0], [1.0, 1.0])
	qm = merge(q1, q2)
	@test qm.xs == [1.0, 2.0, 3.0, 4.0] && qm.ys == [1.0, 2.0, 5.0, 6.0] && qm.ds[1] == 0.0 && qm.ds[end] == 1.0
	qt = merge(q1, PWL([1.0 + 1e-9, 2.0], [5.0, 6.0], [1.0, 1.0]))
	@test qt.n == 2 && qt.ys == [5.0, 6.0]

	# Contract violations.
	@test_throws DomainError PWL(Float64[], Float64[], [0.0, 1.0])
	@test_throws DomainError PWL([1.0], [1.0], [0.0, 1.0])
	@test_throws DomainError PWL([2.0, 1.0], [1.0, 2.0], [0.0, 1.0])
	@test_throws DomainError PWL([1.0, 1.0], [1.0, 2.0], [0.0, 1.0])
	@test_throws DomainError PWL([1.0, 2.0], [1.0, 2.0, 3.0], [0.0, 1.0])
	@test_throws DomainError PWL([1.0, 2.0], [1.0, 2.0], [0.0])
	@test_throws DomainError PWL([1.0, 2.0], 1.0, [0.0, 1.0])
end


# Check inverse trig AD derivatives (chain rule)
@testset "AutoDiff (Inverse Trig)" begin
	x = AD{Float64}(0.5, 3.0)
	# asin'(u) = 1/sqrt(1-u^2), chain rule: d = 3.0 / sqrt(1 - 0.25)
	@test asin(x) ≈ AD{Float64}(asin(0.5), 3.0 / sqrt(0.75)) rtol=TOL

	# acos'(u) = -1/sqrt(1-u^2), chain rule: d = -3.0 / sqrt(0.75)
	@test acos(x) ≈ AD{Float64}(acos(0.5), -3.0 / sqrt(0.75)) rtol=TOL

	# atan'(u) = 1/(1+u^2), chain rule: d = 3.0 / (1 + 0.25)
	@test atan(x) ≈ AD{Float64}(atan(0.5), 3.0 / 1.25) rtol=TOL

	# acot'(u) = -1/(1+u^2), chain rule: d = -3.0 / 1.25
	@test acot(x) ≈ AD{Float64}(acot(0.5), -3.0 / 1.25) rtol=TOL

	x = AD{Float64}(2.0, 3.0)
	# acsc'(u) = -1/(u*sqrt(u^2-1)), chain rule: d = -3.0 / (2*sqrt(3))
	@test acsc(x) ≈ AD{Float64}(acsc(2.0), -3.0 / (2.0 * sqrt(3.0))) rtol=TOL

	# asec'(u) = 1/(|u|*sqrt(u^2-1)), chain rule: d = 3.0 / (2*sqrt(3))
	@test asec(x) ≈ AD{Float64}(asec(2.0), 3.0 / (2.0 * sqrt(3.0))) rtol=TOL
end


# Check trig functions at previously rejected valid inputs
@testset "AutoDiff (Trig Domain)" begin
	# tan(0) should work (was rejected by old mod-based check)
	x = AD{Float64}(0.0, 1.0)
	@test tan(x) ≈ AD{Float64}(0.0, 1.0) rtol=TOL

	# sec(0) = 1, sec'(0) = sec(0)*tan(0) = 0
	@test sec(x) ≈ AD{Float64}(1.0, 0.0) rtol=TOL

	# tan(π) ≈ 0
	x = AD{Float64}(Float64(π), 1.0)
	@test abs(tan(x).v) < 1.0e-12
end


# Check UtilFunc: sigmoid, relu, softmax
@testset "UtilFunc (Activations)" begin
	# sigmoid1: σ(0) = 0.5, σ'(0) = 0.25
	x = AD{Float64}(0.0, 1.0)
	@test sigmoid1(x) ≈ AD{Float64}(0.5, 0.25) rtol=TOL

	# sigmoid2 (tanh): tanh(0) = 0, tanh'(0) = 1
	@test sigmoid2(x) ≈ AD{Float64}(0.0, 1.0) rtol=TOL

	# sigmoid3 (atan): atan(0) = 0, atan'(0) = 1
	@test sigmoid3(x) ≈ AD{Float64}(0.0, 1.0) rtol=TOL

	# relu: relu(1) = 1 with derivative 1; relu(-1) = -1 with derivative 0
	x_pos = AD{Float64}(1.0, 1.0)
	@test relu(x_pos) ≈ AD{Float64}(1.0, 1.0) rtol=TOL

	x_neg = AD{Float64}(-1.0, 1.0)
	@test relu(x_neg) ≈ AD{Float64}(-1.0, 0.0) rtol=TOL

	# relur: same value as relu; the derivative boundary is random but reproducible with an rng.
	@test relur(AD(-0.05, 1.0); rng=MersenneTwister(1)) == relur(AD(-0.05, 1.0); rng=MersenneTwister(1))
	@test relur(AD(1.0, 1.0)) == AD(1.0, 1.0) && relur(AD(-1.0, 1.0)) == AD(-1.0, 0.0)

	# softmax: on equal inputs, should return uniform distribution
	xs = [1.0, 1.0, 1.0]
	sm = softmax(xs)
	@test all(abs.(sm .- 1.0/3.0) .< 1.0e-10)
	@test softmax([1.0, 2.0]) ≈ [exp(1.0), exp(2.0)] ./ (exp(1.0) + exp(2.0))
	@test softmax([1.0, 2.0], 2.0) ≈ [exp(0.5), exp(1.0)] ./ (exp(0.5) + exp(1.0))
	sma = softmax([AD(1.0, 1.0), AD(2.0, 0.0)])
	@test [v.v for v in sma] ≈ softmax([1.0, 2.0])
	@test sma[1].d ≈ sma[1].v * (1 - sma[1].v) && sma[2].d ≈ -sma[1].d
	@test_throws DomainError softmax([1.0, 2.0], 0.0)
	@test_throws DomainError softmax(Float64[])

	# L1 for numbers and AD values.
	@test L1([-1.0, 2.0, -3.0]) == 6.0
	@test L1([AD(-1.0, 1.0), AD(2.0, 1.0)]) == AD(3.0, 0.0)
end


# Check DNN construction, forward pass, loss, fit
@testset "DNNS (DNN Construction and Forward)" begin
	# Build a simple 2→3→1 network
	M1 = [1.0 0.0; 0.0 1.0; 1.0 1.0]
	b1 = [0.0, 0.0, 0.0]
	l1 = DLayer(M1, b1, sigmoid1)
	@test l1.dims == (3, 2)

	M2 = [1.0 1.0 1.0]
	b2 = [0.0]
	l2 = DLayer(M2, b2, sigmoid1)
	@test l2.dims == (1, 3)

	dnn = DNN([l1, l2])

	# Forward pass should produce a 1-element vector
	out = dnn([0.5, 0.5])
	@test length(out) == 1
	@test out[1].v > 0.0  # sigmoid outputs are positive
	@test out[1] ≈ sigmoid1(AD(sum(sigmoid1.(AD.([0.5, 0.5, 1.0]))).v))
	@test dnn([1, 1]) == dnn([1.0, 1.0])            # integer input
	@test dnn(view([0.5, 0.5, 9.0], 1:2)) == out   # views

	# Element types: a Float32 layer accepts Float64 input and keeps Float32.
	l32 = DLayer(Float32[1 0; 0 1], Float32[0, 0], relu)
	@test l32([1.0, 2.0]) == AD{Float32}[AD(1.0f0, 0.0f0), AD(2.0f0, 0.0f0)]
	@test DLayer([1 0; 0 1], [0.5, 0.5], relu) isa DLayer{Float64}

	# Loss should be non-negative
	X = [0.0 1.0; 0.0 1.0]
	Y = [0.5 0.8]
	ls = loss(dnn, X, Y)
	@test ls.v >= 0.0
	@test ls.v ≈ ((dnn([0.0, 0.0])[1].v - 0.5)^2 + (dnn([1.0, 1.0])[1].v - 0.8)^2) / 2

	# Contract violations.
	@test_throws DomainError DLayer([1.0 0.0; 0.0 1.0], [0.0], relu)
	@test_throws DomainError DNN([l2, l1])
	@test_throws DomainError DNN(DLayer{Float64}[])
	@test_throws DomainError dnn([1.0, 2.0, 3.0])
	@test_throws DomainError loss(dnn, X, [0.5 0.8 0.1])
	@test_throws DomainError loss(dnn, X, [0.5 0.8; 0.1 0.2])
end

@testset "DNNS (fit)" begin
	# Fit a 1→2→1 network to y = 2x + 1 on a few points; the loss must decrease and nothing is printed.
	Random.seed!(11)
	l1 = DLayer(rand(2, 1), rand(2), sigmoid2)
	l2 = DLayer(rand(1, 2), rand(1), relu)
	dnn = DNN([l1, l2])
	X = reshape(collect(0.0:0.1:1.0), 1, :)
	Y = 2.0 .* X .+ 1.0
	l0 = loss(dnn, X, Y).v
	res = mktemp() do path, io
		redirect_stdout(io) do
			fit(dnn, X, Y; N=200, μ=0.05)
		end
	end
	@test res.loss < l0
	@test res.loss ≈ loss(dnn, X, Y).v
	@test res.iterations == 200 && !res.converged
	@test all(p -> p.d == 0.0, dnn.layers[1].M)   # parameters are left as constants

	# Convergence stops early and reports the iteration count.
	res2 = fit(dnn, X, Y; N=5000, μ=0.05, relerr=1e-3)
	@test res2.converged && res2.iterations < 5000
	@test fit(dnn, X, Y; N=0).iterations == 0
	@test_throws DomainError fit(dnn, X, Y[:, 1:3])
end

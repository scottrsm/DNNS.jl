using Test

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
end

# Check PWLF module
@testset "PWLF (PWLF Calculations)" begin
	p1 = PWL([1.0, 3.0, 4.0, 6.0], [2.0, 4.0, 6.0, 10.0], [0.0, 1.0])
	p2 = PWL([1.0, 3.0, 4.0, 6.0], [3.0, 30.0, 3.0, 10.0], [-2.0, 5.0])
	pm = merge(p1, p2)
	ps = PWLF.smooth(pm, 1.5)

	@test pm ≈ PWL([1.0, 3.0, 4.0, 6.0], [3.0, 30.0, 3.0, 10.0], [-2.0, 5.0])  rtol=TOL 
	@test ps ≈ PWL([1.0, 3.5, 6.0], [3.0, 16.5, 10.0], [-2.0, 5.0])            rtol=TOL
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

	# softmax: on equal inputs, should return uniform distribution
	xs = [1.0, 1.0, 1.0]
	sm = softmax(xs)
	@test all(abs.(sm .- 1.0/3.0) .< 1.0e-10)
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

	# Loss should be non-negative
	X = [0.0 1.0; 0.0 1.0]
	Y = [0.5 0.8]
	ls = loss(dnn, X, Y)
	@test ls.v >= 0.0
end


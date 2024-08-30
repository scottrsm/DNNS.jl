using Test

using DNNS

include("../src/AutoDiff.jl")
import ..AutoDiff: AD

include("../src/PWLF.jl")
import ..PWLF

include("../src/UtilFunc.jl")
import ..UtilFunc

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
end

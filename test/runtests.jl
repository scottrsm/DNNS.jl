using Test

using DNNS

@testset "DNNS (Fidelity)" begin
    @test length(detect_ambiguities(DNNS)) == 0
end


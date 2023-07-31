using Test
using LowRankApprox
using LinearAlgebra, Random, SparseArrays

using Aqua
@testset "Project quality" begin
    Aqua.test_all(LowRankApprox, ambiguities=false)
end

Random.seed!(0)


include("linop.jl")
include("permute.jl")
include("snorm.jl")
include("trapezoidal.jl")

println()

include("sketch.jl")
include("id.jl")
include("pheig.jl")
include("pqr.jl")
include("prange.jl")
include("psvd.jl")
include("cur.jl")

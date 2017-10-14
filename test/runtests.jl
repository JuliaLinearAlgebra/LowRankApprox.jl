if VERSION ≤ v"0.7.0-DEV.1775"
    using Base.Test
else
    using Test
end
using LowRankApprox

srand(0)

include("linop.jl")
include("permute.jl")
include("snorm.jl")
include("trapezoidal.jl")

println()

include("cur.jl")
include("id.jl")
include("pheig.jl")
include("pqr.jl")
include("prange.jl")
include("psvd.jl")
include("sketch.jl")

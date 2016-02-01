using Base.Test
using LowRankApprox

include("permute.jl")
include("trapezoidal.jl")

println()

include("cur.jl")
include("id.jl")
include("pheig.jl")
include("pqr.jl")
include("prange.jl")
include("psvd.jl")
include("sketch.jl")
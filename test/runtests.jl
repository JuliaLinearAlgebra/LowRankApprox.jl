using Compat
using Compat.Test
using LowRankApprox
using Compat.LinearAlgebra, Compat.Random, Compat.SparseArrays

if VERSION < v"0.7"
    srand(0)
else
    Random.seed!(0)
end

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
include("lowrankmatrix.jl")

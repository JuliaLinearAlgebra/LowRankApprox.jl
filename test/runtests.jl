using Compat
using Compat.Test
using LowRankApprox
using Compat.LinearAlgebra, Compat.Random, Compat.SparseArrays

srand(0)

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

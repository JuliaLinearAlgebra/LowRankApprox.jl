#= src/LowRankApprox.jl
=#

module LowRankApprox

importall Base
using Base.LinAlg: BlasFloat, BlasInt, chksquare, chkstride1

export

  # LowRankApprox.jl
  LRAOptions,

  # cur.jl
  AbstractCURPackedU,
  CURPackedU,
  HermitianCURPackedU,
  SymmetricCURPackedU,
  AbstractCUR,
  CUR,
  HermitianCUR,
  SymmetricCUR,
  curfact, curfact!,
  cur, cur!,

  # id.jl
  IDPackedV,
  ID,
  idfact, idfact!,
  id, id!,

  # linop.jl
  AbstractLinearOperator,
  LinearOperator,
  HermitianLinearOperator,

  # matrixlib.jl
  matrixlib,

  # permute.jl
  PermutationMatrix,
  RowPermutation,
  ColumnPermutation,

  # pheig.jl
  HermitianPartialEigen,
  pheigfact,
  pheig,
  pheigvals,

  # pqr.jl
  PartialQRFactors,
  PartialQR,
  pqrfact, pqrfact!,
  pqr, pqr!,

  # prange.jl
  prange, prange!,

  # psvd.jl
  PartialSVD,
  psvdfact,
  psvd,
  psvdvals,

  # sketch.jl
  SketchMatrix,
  RandomGaussian,
  RandomSubset,
  SRFT,
  SparseRandomGaussian,
  sketch,
  sketchfact,

  # snorm.jl
  snorm,
  snormdiff,

  # trapezoidal.jl
  Trapezoidal,
  LowerTrapezoidal,
  UpperTrapezoidal

# common

type LRAOptions
  atol::Float64
  nb::Int
  pqrfact_retval::ASCIIString
  rank::Int
  rrqr_delta::Float64
  rrqr_niter::Int
  rtol::Float64
  sketch::Symbol
  sketch_randn_niter::Int
  sketchfact_adap::Bool
  sketchfact_randn_samp::Function
  sketchfact_srft_samp::Function
  sketchfact_sub_samp::Function
  snorm_niter::Int
  verb::Bool
end

function LRAOptions{T}(::Type{T}; args...)
  opts = LRAOptions(
    0,              # atol
    32,             # nb
    "qr",           # pqrfact_retval
    -1,             # rank
    -1,             # rrqr_delta
    -1,             # rrqr_niter
    5*eps(real(T)), # rtol
    :randn,         # sketch
    0,              # sketch_randn_niter
    true,           # sketchfact_adap
    n -> n + 8,     # sketchfact_randn_samp
    n -> n + 8,     # sketchfact_srft_samp
    n -> 4*n + 8,   # sketchfact_sub_samp
    32,             # snorm_niter
    true,           # verb
  )
  for (key, value) in args
    setfield!(opts, key, value)
  end
  opts
end
LRAOptions(; args...) = LRAOptions(Float64; args...)

function copy(opts::LRAOptions; args...)
  opts_ = LRAOptions()
  for field in fieldnames(opts)
    setfield!(opts_, field, getfield(opts, field))
  end
  for (key, value) in args
    setfield!(opts_, key, value)
  end
  opts_
end

function chkopts!(opts::LRAOptions)
  opts.atol >= 0 || throw(ArgumentError("atol"))
  opts.nb > 0 || throw(ArgumentError("nb"))
  opts.rtol >= 0 || throw(ArgumentError("rtol"))
  opts.sketch in (:none, :randn, :sprn, :srft, :sub) ||
    throw(ArgumentError("sketch"))
  opts.pqrfact_retval = lowercase(opts.pqrfact_retval)
end
function chkopts!(opts::LRAOptions, A)
  chkopts!(opts)
  if typeof(A) <: AbstractLinOp && opts.sketch != :randn
    warn("invalid sketch method; using \"randn\"")
    opts.sketch = :randn
  end
end

chktrans(trans::Symbol) = trans in (:n, :c) || throw(ArgumentError("trans"))

# source files

include("lapack.jl")
include("linop.jl")
include("matrixlib.jl")
include("permute.jl")
include("snorm.jl")
include("trapezoidal.jl")
include("util.jl")

include("cur.jl")
include("id.jl")
include("pheig.jl")
include("pqr.jl")
include("prange.jl")
include("psvd.jl")
include("sketch.jl")

end  # module
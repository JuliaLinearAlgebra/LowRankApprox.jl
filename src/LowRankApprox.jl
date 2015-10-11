#= src/LowRankApprox.jl
=#

module LowRankApprox

importall Base
import Base.LinAlg: BlasFloat, BlasInt, chksquare, chkstride1

export

  # LowRankApprox.jl
  LRAOptions,

  # cur.jl
  AbstractCURPackedU,
  CURPackedU,
  HermitianCURPackedU,
  AbstractCUR,
  CUR,
  HermitianCUR,
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

  # peig.jl
  AbstractPartialEigen,
  PartialEigen,
  HermitianPartialEigen,
  peigfact,
  peig,
  peigvals,

  # permute.jl
  PermutationMatrix,
  RowPermutation,
  ColumnPermutation,

  # pqr.jl
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
  peig_vecs::Symbol
  rank::Int
  rtol::Float64
  sketch::Symbol
  sketch_randn_niter::Int
  sketch_randn_samp::Int
  sketch_srft_samp::Int
  sketch_sub_samp::Int
  snorm_info::Bool
  snorm_niter::Int
end

LRAOptions(;
    atol::Real=0.,
    nb::Integer=32,
    peig_vecs::Symbol=:right,
    rank::Integer=-1,
    rtol::Real=default_rtol(Float64),
    sketch::Symbol=:randn,
    sketch_randn_niter::Integer=0,
    sketch_randn_samp::Integer=8,
    sketch_srft_samp::Integer=8,
    sketch_sub_samp::Integer=6,
    snorm_info::Bool=false,
    snorm_niter::Integer=32,
    ) =
  LRAOptions(
    atol,
    nb,
    peig_vecs,
    rank,
    rtol,
    sketch,
    sketch_randn_niter,
    sketch_randn_samp,
    sketch_srft_samp,
    sketch_sub_samp,
    snorm_info,
    snorm_niter,
    )

function copy(opts::LRAOptions; args...)
  opts_ = LRAOptions(
    opts.atol,
    opts.nb,
    opts.peig_vecs,
    opts.rank,
    opts.rtol,
    opts.sketch,
    opts.sketch_randn_niter,
    opts.sketch_randn_samp,
    opts.sketch_srft_samp,
    opts.sketch_sub_samp,
    opts.snorm_info,
    opts.snorm_niter,
    )
  for (key, value) in args
    setfield!(opts_, key, value)
  end
  opts_
end

function chkopts(opts)
  opts.atol >= 0 || throw(ArgumentError("atol"))
  opts.nb > 0 || throw(ArgumentError("nb"))
  opts.peig_vecs in (:left, :right) || throw(ArgumentError("peig_vecs"))
  opts.rtol >= 0 || throw(ArgumentError("rtol"))
  opts.sketch in (:none, :randn, :sprn, :srft, :sub) ||
    throw(ArgumentError("sketch"))
  opts.sketch_randn_samp >= 0 || throw(ArgumentError("sketch_randn_samp"))
  opts.sketch_srft_samp >= 0 || throw(ArgumentError("sketch_srft_samp"))
  opts.sketch_sub_samp > 0 || throw(ArgumentError("sketch_sub_samp"))
end
function chkopts(A, opts::LRAOptions)
  chkopts(opts)
  if typeof(A) <: AbstractLinOp && opts.sketch != :randn
    warn("invalid sketch method; using \"randn\"")
    opts = copy(opts, sketch=:randn)
  end
  opts
end

chktrans(trans::Symbol) = trans in (:n, :c) || throw(ArgumentError("trans"))

default_rtol{T}(::Type{T}) = 5*eps(real(T))

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
include("peig.jl")
include("pqr.jl")
include("prange.jl")
include("psvd.jl")
include("sketch.jl")

end  # module
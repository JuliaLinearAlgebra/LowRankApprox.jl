#= src/LowRankApprox.jl
=#

module LowRankApprox

importall Base
import Base.LinAlg: BlasFloat, BlasInt, QRCompactWY, chksquare, chkstride1

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
end

LRAOptions(;
    atol::Real=0.,
    nb::Integer=32,
    pqrfact_retval::ASCIIString="qr",
    rank::Integer=-1,
    rrqr_delta::Real=-1,
    rrqr_niter::Integer=-1,
    rtol::Real=default_rtol(Float64),
    sketch::Symbol=:randn,
    sketch_randn_niter::Integer=0,
    sketchfact_adap::Bool=true,
    sketchfact_randn_samp::Function=(n -> n + 8),
    sketchfact_srft_samp::Function=(n -> n + 8),
    sketchfact_sub_samp::Function=(n -> 4*n + 8),
    snorm_niter::Integer=32,
    ) =
  LRAOptions(
    atol,
    nb,
    pqrfact_retval,
    rank,
    rrqr_delta,
    rrqr_niter,
    rtol,
    sketch,
    sketch_randn_niter,
    sketchfact_adap,
    sketchfact_randn_samp,
    sketchfact_srft_samp,
    sketchfact_sub_samp,
    snorm_niter,
    )

function copy(opts::LRAOptions; args...)
  opts_ = LRAOptions(
    opts.atol,
    opts.nb,
    opts.pqrfact_retval,
    opts.rank,
    opts.rrqr_delta,
    opts.rrqr_niter,
    opts.rtol,
    opts.sketch,
    opts.sketch_randn_niter,
    opts.sketchfact_adap,
    opts.sketchfact_randn_samp,
    opts.sketchfact_srft_samp,
    opts.sketchfact_sub_samp,
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
  opts.rtol >= 0 || throw(ArgumentError("rtol"))
  opts.sketch in (:none, :randn, :sprn, :srft, :sub) ||
    throw(ArgumentError("sketch"))
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
include("pheig.jl")
include("pqr.jl")
include("prange.jl")
include("psvd.jl")
include("sketch.jl")

end  # module
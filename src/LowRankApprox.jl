#= src/LowRankApprox.jl
=#

module LowRankApprox

importall Base
import Base.LinAlg: BlasComplex, BlasFloat, BlasInt, BlasReal, chkstride1

export

  # LowRankApprox.jl
  LRAOptions,

  # linop.jl
  AbstractLinearOperator,
  LinearOperator,
  HermitianLinearOperator,

  # permute.jl
  PermutationMatrix,
  RowPermutation,
  ColumnPermutation,

  # pqr.jl
  pqr, pqr!,
  pqrfact, pqrfact!,

  # rrange.jl

  # sketch.jl
  SketchMatrix,
  RandomGaussian,
  RandomSubset,
  SRFT,
  sketch,
  sketchfact,

  # snorm.jl
  snorm,
  snormdiff,

  # trapezoidal.jl
  Trapezoidal,
  LowerTrapezoidal,
  UpperTrapezoidal

#

type LRAOptions
  atol::Float64
  nb::Int
  rank::Int
  rtol::Float64
  sketch::Symbol
  sketch_randn_niter::Int
  sketch_randn_samp::Int
  sketch_srft_samp::Int
  sketch_subs_samp::Int
  snorm_info::Bool
  snorm_niter::Int
end

LRAOptions(;
    atol::Real=0.,
    nb::Integer=32,
    rank::Integer=-1,
    rtol::Real=eps(),
    sketch::Symbol=:none,
    sketch_randn_niter::Integer=0,
    sketch_randn_samp::Integer=8,
    sketch_srft_samp::Integer=8,
    sketch_subs_samp::Integer=6,
    snorm_info::Bool=false,
    snorm_niter::Integer=32) =
  LRAOptions(
    atol,
    nb,
    rank,
    rtol,
    sketch,
    sketch_randn_niter,
    sketch_randn_samp,
    sketch_srft_samp,
    sketch_subs_samp,
    snorm_info,
    snorm_niter)

copy(opts::LRAOptions) =
  LRAOptions(
    opts.atol,
    opts.nb,
    opts.rank,
    opts.rtol,
    opts.sketch,
    opts.sketch_randn_niter,
    opts.sketch_randn_samp,
    opts.sketch_srft_samp,
    opts.sketch_subs_samp,
    opts.snorm_info,
    opts.snorm_niter)

function chkopts(opts)
  opts.atol >= 0 || throw(ArgumentError("atol"))
  opts.nb > 0 || throw(ArgumentError("nb"))
  opts.rtol >= 0 || throw(ArgumentError("rtol"))
  opts.sketch in (:none, :randn, :srft, :subs) || throw(ArgumentError("sketch"))
  opts.sketch_randn_samp >= 0 || throw(ArgumentError("sketch_randn_samp"))
  opts.sketch_srft_samp >= 0 || throw(ArgumentError("sketch_srft_samp"))
  opts.sketch_subs_samp > 0 || throw(ArgumentError("sketch_subs_samp"))
end

#
include("lapack.jl")
include("linop.jl")
include("permute.jl")
include("pqr.jl")
include("rrange.jl")
include("sketch.jl")
include("snorm.jl")
include("trapezoidal.jl")
include("util.jl")

end  # module
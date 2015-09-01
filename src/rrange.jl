#= src/rrange.jl

References:

  B. Engquist, L. Ying. A fast directional algorithm for high frequency
    acoustic scattering in two dimensions. Commun. Math. Sci. 7 (2): 327-345,
    2009.

  N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM Rev. 53 (2): 217-288, 2011.
=#

function rrange(trans::Symbol, A::AbstractMatOrLinOp, opts::LRAOptions)
  chkopts(opts)
  opts = sketch_chkopts(typeof(A), opts)
  rrange_chkargs(trans)
  if opts.sketch == :subs
    return rrange_subs(trans, A, opts)
  end
  F = sketchfact(:right, trans, A, opts)
  F[:Q]
end
function rrange(trans::Symbol, A::AbstractMatOrLinOp, rank_or_rtol::Real)
  opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol, sketch=:randn)
                           : LRAOptions(rank=rank_or_rtol, sketch=:randn))
  rrange(trans, A, opts)
end
rrange{T}(trans::Symbol, A::AbstractMatOrLinOp{T}) =
  rrange(trans, A, default_rtol(T))
rrange(trans::Symbol, A, args...) = rrange(trans, LinOp(A), args...)

function rrange_subs{T}(trans::Symbol, A::AbstractMatrix{T}, opts::LRAOptions)
  F = sketchfact(:left, trans, A, opts)
  idx = F[:p][1:F[:k]]
  if trans == :n
    B = A[:,idx]
  else
    n = size(A, 2)
    B = Array(T, n, F[:k])
    for j = 1:F[:k], i = 1:n
      B[i,j] = conj(A[F[:p][j],i])
    end
  end
  orthcols!(B)
end

rrange_chkargs(trans::Symbol) =
  trans in (:n, :c) || throw(ArgumentError("trans"))
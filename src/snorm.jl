#= src/snorm.jl

References:

  J. Dixon. Estimating extremal eigenvalues and condition numbers of matrices.
    SIAM J. Numer. Anal. 20 (4): 812-814, 1983.

  J. Kuczynski, H. Wozniakowski. Estimating the largest eigenvalue by the power
    and Lanczos algorithms with a random start. SIAM J. Matrix Anal. Appl. 13
    (4), 1094-1122, 1992.
=#

type ConvergenceInfo{T<:Real}
  iterates::Vector{T}
  neval::Int
  niter::Int
  converged::Bool
end

# spectral norm
function snorm{T}(A::AbstractLinOp{T}, opts::LRAOptions)
  chkopts(opts)
  m, n      = size(A)
  isherm    = ishermitian(A)
  xn        = _randn(T, n)
  xm        = Array(T, m)
  xnrm      = vecnorm(xn)
  s         = [real(one(T))]
  t         = 0
  neval     = 0
  niter     = 0
  converged = true
  while s[end] > 0 && abs(s[end] - t) > max(opts.atol, t*opts.rtol)
    niter += 1
    xn /= xnrm
    if isherm
      A_mul_B!(xm, A, xn)
      copy!(xn, xm)
      neval += 1
    else
       A_mul_B!(xm, A, xn)
      Ac_mul_B!(xn, A, xm)
      neval += 2
    end
    xnrm = vecnorm(xn)
    t = s[end]
    push!(s, isherm ? xnrm : sqrt(xnrm))
    if niter == opts.snorm_niter
      warn("maximum number of iterations ($niter) reached")
      converged = false
      break
    end
  end
  if opts.snorm_info  ConvergenceInfo(s, neval, niter, converged)
  else                s[end]
  end
end
function snorm(A::AbstractLinOp, niter_or_rtol::Real)
  opts = (niter_or_rtol < 1 ? LRAOptions(       rtol=niter_or_rtol)
                            : LRAOptions(snorm_niter=niter_or_rtol))
  snorm(A, opts)
end
snorm{T}(A::AbstractLinOp{T}) = snorm(A, default_rtol(T))
snorm(A, args...) = snorm(LinOp(A), args...)

# spectral norm difference
snormdiff{T}(A::AbstractLinOp{T}, B::AbstractLinOp{T}, args...) =
  snorm(A - B, args...)
snormdiff(A, B, args...) = snormdiff(LinOp(A), LinOp(B), args...)
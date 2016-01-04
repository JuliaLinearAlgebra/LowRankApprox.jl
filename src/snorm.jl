#= src/snorm.jl

References:

  J. Dixon. Estimating extremal eigenvalues and condition numbers of matrices.
    SIAM J. Numer. Anal. 20 (4): 812-814, 1983.

  J. Kuczynski, H. Wozniakowski. Estimating the largest eigenvalue by the power
    and Lanczos algorithms with a random start. SIAM J. Matrix Anal. Appl. 13
    (4), 1094-1122, 1992.
=#

# spectral norm
function snorm{T}(A::AbstractLinOp{T}, opts::LRAOptions=LRAOptions(T); args...)
  opts = copy(opts; args...)
  chkopts!(opts)
  m, n   = size(A)
  isherm = ishermitian(A)
  xn     = crandn(T, n)
  xm     = Array(T, m)
  xnrm   = vecnorm(xn)
  s      = one(real(T))
  t      = 0
  niter  = 0
  while s > 0 && abs(s - t) > max(opts.atol, t*opts.rtol)
    if niter == opts.snorm_niter
      warn("iteration limit ($niter) reached in spectral norm estimation")
      break
    end
    niter += 1
    scale!(xn, 1/xnrm)
    if isherm
      A_mul_B!(xm, A, xn)
      copy!(xn, xm)
    else
       A_mul_B!(xm, A, xn)
      Ac_mul_B!(xn, A, xm)
    end
    xnrm = vecnorm(xn)
    t = s
    s = isherm ? xnrm : sqrt(xnrm)
  end
  s
end
snorm(A, args...; kwargs...) = snorm(LinOp(A), args...; kwargs...)

# spectral norm difference
snormdiff{T}(A::AbstractLinOp{T}, B::AbstractLinOp{T}, args...; kwargs...) =
  snorm(A - B, args...; kwargs...)
snormdiff(A, B, args...; kwargs...) =
  snormdiff(LinOp(A), LinOp(B), args...; kwargs...)
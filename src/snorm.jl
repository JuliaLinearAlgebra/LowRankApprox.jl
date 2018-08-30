#= src/snorm.jl

References:

  J. Dixon. Estimating extremal eigenvalues and condition numbers of matrices.
    SIAM J. Numer. Anal. 20 (4): 812-814, 1983.

  J. Kuczynski, H. Wozniakowski. Estimating the largest eigenvalue by the power
    and Lanczos algorithms with a random start. SIAM J. Matrix Anal. Appl. 13
    (4), 1094-1122, 1992.
=#

# spectral norm
  function snorm(A::AbstractLinOp{T}, opts::LRAOptions=LRAOptions(T); args...) where T
    opts = isempty(args) ? opts : copy(opts; args...)
    m, n   = size(A)
    isherm = ishermitian(A)
    xn     = crandn(T, n)
    xm     = Array{T}(undef, m)
    xnrm   = norm(xn)
    s      = one(real(T))
    t      = 0
    niter  = 0
    while s > 0 && abs(s - t) > max(opts.atol, t*opts.rtol)
      if niter == opts.snorm_niter
        opts.verb && warn("iteration limit ($(opts.snorm_niter)) reached")
        break
      end
      niter += 1
      rmul!(xn, 1/xnrm)
      if isherm
        mul!(xm, A, xn)
        copyto!(xn, xm)
      else
         mul!(xm, A, xn)
        mul!(xn, A', xm)
      end
      xnrm = norm(xn)
      t = s
      s = isherm ? xnrm : sqrt(xnrm)
    end
    s
  end

snorm(A, args...; kwargs...) = snorm(LinOp(A), args...; kwargs...)

# spectral norm difference
snormdiff(A::AbstractLinOp{T}, B::AbstractLinOp{T}, args...; kwargs...) where {T} =
  snorm(A - B, args...; kwargs...)
snormdiff(A, B, args...; kwargs...) =
  snormdiff(LinOp(A), LinOp(B), args...; kwargs...)

#= src/prange.jl

References:

  B. Engquist, L. Ying. A fast directional algorithm for high frequency
    acoustic scattering in two dimensions. Commun. Math. Sci. 7 (2): 327-345,
    2009.

  N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM Rev. 53 (2): 217-288, 2011.
=#

for sfx in ("", "!")
  f = symbol("prange", sfx)
  g = symbol("pqrfact", sfx)
  @eval begin
    function $f{T}(trans::Symbol, A::AbstractMatOrLinOp{T}, opts::LRAOptions)
      chkopts(opts)
      opts = prange_chkopts(typeof(A), opts)
      prange_chkargs(trans)
      if trans == :nc
        if opts.sketch == :none
          Fc =  pqrfact(A , opts)
          Fr = pqrfact!(A', opts)
        else
          Fc = sketchfact(:right, :n, A, opts)
          Fr = sketchfact(:right, :c, A, opts)
        end
        kc = Fc[:k]
        kr = Fr[:k]
        B = Array(T, size(A,1), kc+kr)
        B[:,   1:kc   ] = Fc[:Q]
        B[:,kc+1:kc+kr] = Fr[:Q]
        Rc = sub(Fc.R, 1:kc, 1:kc)
        Rr = sub(Fr.R, 1:kr, 1:kr)
        BLAS.trmm!('R', 'U', 'N', 'N', one(T), Rc, sub(B,:,   1:kc   ))
        BLAS.trmm!('R', 'U', 'N', 'N', one(T), Rr, sub(B,:,kc+1:kc+kr))
        Q = pqrfact_lapack!(B, opts)[:Q]
      else
        if     opts.sketch == :none  Q = $g(trans == :n ? A : A', opts)[:Q]
        elseif opts.sketch == :subs  Q = prange_subs(trans, A, opts)
        else                         Q = sketchfact(:right, trans, A, opts)[:Q]
        end
      end
      Q
    end
    function $f(trans::Symbol, A::AbstractMatOrLinOp, rank_or_rtol::Real)
      opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol, sketch=:randn)
                               : LRAOptions(rank=rank_or_rtol, sketch=:randn))
      $f(trans, A, opts)
    end
    $f{T}(trans::Symbol, A::AbstractMatOrLinOp{T}) =
      $f(trans, A, default_rtol(T))
    $f(trans::Symbol, A, args...) = $f(trans, LinOp(A), args...)
  end
end

function prange_subs{T}(trans::Symbol, A::AbstractMatrix{T}, opts::LRAOptions)
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

function prange_chkopts{T}(::Type{T}, opts::LRAOptions)
  if T <: AbstractLinOp && opts.sketch != :randn
    warn("invalid sketch method; using \"randn\"")
    opts = copy(opts, sketch=:randn)
  end
  opts
end

prange_chkargs(trans::Symbol) =
  trans in (:n, :c, :nc) || throw(ArgumentError("trans"))
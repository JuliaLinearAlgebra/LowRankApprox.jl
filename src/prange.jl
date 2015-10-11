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
      opts = chkopts(A, opts)
      prange_chktrans(trans)
      if trans == :b
        if opts.sketch == :none
          Fr = pqrfact!(A', opts)
          Fc =       $g(A , opts)
        else
          Fr = sketchfact(:right, :c, A, opts)
          Fc = sketchfact(:right, :n, A, opts)
        end
        kr = Fr[:k]
        kc = Fc[:k]
        B = Array(T, size(A,1), kr+kc)
        B[:,   1:kr   ] = Fr[:Q]
        B[:,kr+1:kr+kc] = Fc[:Q]
        Rr = sub(Fr.R, 1:kr, 1:kr)
        Rc = sub(Fc.R, 1:kc, 1:kc)
        BLAS.trmm!('R', 'U', 'N', 'N', one(T), Rr, sub(B,:,   1:kr   ))
        BLAS.trmm!('R', 'U', 'N', 'N', one(T), Rc, sub(B,:,kr+1:kr+kc))
        return pqrfact_lapack!(B, opts)[:Q]
      else
        if opts.sketch == :none
          if trans == :n  Q =       $g(A , opts)[:Q]
          else            Q = pqrfact!(A', opts)[:Q]
          end
        elseif opts.sketch == :sub  Q = prange_sub(trans, A, opts)
        else                        Q = sketchfact(:right, trans, A, opts)[:Q]
        end
        Q
      end
    end
    function $f(trans::Symbol, A::AbstractMatOrLinOp, rank_or_rtol::Real)
      opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol)
                               : LRAOptions(rank=rank_or_rtol))
      $f(trans, A, opts)
    end
    $f{T}(trans::Symbol, A::AbstractMatOrLinOp{T}) =
      $f(trans, A, default_rtol(T))
    $f(trans::Symbol, A, args...) = $f(trans, LinOp(A), args...)
    $f(A, args...) = $f(:n, A, args...)
  end
end

function prange_sub{T}(trans::Symbol, A::AbstractMatrix{T}, opts::LRAOptions)
  F = sketchfact(:left, trans, A, opts)
  k = F[:k]
  if trans == :n
    B = A[:,F[:p][1:k]]
  else
    n = size(A, 2)
    B = Array(T, n, k)
    for j = 1:k, i = 1:n
      B[i,j] = conj(A[F[:p][j],i])
    end
  end
  orthcols!(B)
end

prange_chktrans(trans::Symbol) =
  trans in (:n, :c, :b) || throw(ArgumentError("trans"))
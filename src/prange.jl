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
  f = Symbol("prange", sfx)
  g = Symbol("pqrfact", sfx)
  @eval begin
    function $f(
        trans::Symbol, A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T);
        args...) where T
      prange_chktrans(trans)
      opts = copy(opts; args...)
      chkopts!(opts, A)
      if trans == :b
        checksquare(A)
        ishermitian(A) && return $f(:n, A, opts)
        opts.pqrfact_retval = "qr"
        if opts.sketch == :none
          Fr = pqrfact!(A', opts)
          Fc =       $g(A , opts)
        else
          Fr = sketchfact(:right, :c, A, opts)
          Fc = sketchfact(:right, :n, A, opts)
        end
        kr = Fr[:k]
        kc = Fc[:k]
        B = Array{T}(undef, size(A,1), kr+kc)
        B[:,   1:kr   ] = Fr[:Q]
        B[:,kr+1:kr+kc] = Fc[:Q]
        Rr = view(Fr.R, 1:kr, 1:kr)
        Rc = view(Fc.R, 1:kc, 1:kc)
        rmul!(view(B,:,   1:kr   ), UpperTriangular(Rr))
        rmul!(view(B,:,kr+1:kr+kc), UpperTriangular(Rc))
        opts.pqrfact_retval="q"
        return pqrfact_backend!(B, opts)[:Q]
      else
        opts.pqrfact_retval = "q"
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
    $f(trans::Symbol, A, args...; kwargs...) =
      $f(trans, LinOp(A), args...; kwargs...)
    $f(A, args...; kwargs...) = $f(:n, A, args...; kwargs...)
  end
end

function prange_sub(trans::Symbol, A::AbstractMatrix{T}, opts::LRAOptions) where T
  F = sketchfact(:left, trans, A, opts)
  k = F[:k]
  if trans == :n
    B = A[:,F[:p][1:k]]
  else
    n = size(A, 2)
    B = Array{T}(undef, n, k)
    @inbounds for j = 1:k, i = 1:n
      B[i,j] = conj(A[F[:p][j],i])
    end
  end
  orthcols!(B)
end

prange_chktrans(trans::Symbol) =
  trans in (:n, :c, :b) || throw(ArgumentError("trans"))

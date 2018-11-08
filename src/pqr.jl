#= src/pqr.jl

References:

  P. Businger, G.H. Golub. Linear least squares solutions by Householder
    transformations. Numer. Math. 7: 269-276, 1965.

  M. Gu, S.C. Eisenstat. Efficient algorithms for computing a strong
    rank-revealing QR factorization. SIAM J. Sci. Comput. 17 (4): 848-869, 1996.
=#

# PartialQRFactors

mutable struct PartialQRFactors
  Q::Nullable{Matrix}
  R::Nullable{Matrix}
  p::Vector{Int}
  k::Int
  T::Nullable{Matrix}
end
const PQRFactors = PartialQRFactors

function getindex(A::PQRFactors, d::Symbol)
  if     d == :P  return ColumnPermutation(A.p)
  elseif d == :Q  return get(A.Q)
  elseif d == :R  return UpperTrapezoidal(get(A.R))
  elseif d == :T  return get(A.T)
  elseif d == :k  return A.k
  elseif d == :p  return A.p
  else            throw(KeyError(d))
  end
end

# PartialQR

mutable struct PartialQR{T} <: Factorization{T}
  Q::Matrix{T}
  R::Matrix{T}
  p::Vector{Int}
end

conj!(A::PartialQR) = PartialQR(conj!(A.Q), conj!(A.R), A.p)
conj(A::PartialQR) = PartialQR(conj(A.Q), conj(A.R), A.p)

convert(::Type{PartialQR{T}}, A::PartialQR) where {T} =
  PartialQR(convert(Array{T}, A.Q), convert(Array{T}, A.R), A.p)
convert(::Factorization{T}, A::PartialQR) where {T} = convert(PartialQR{T}, A)
convert(::Type{Array}, A::PartialQR) = Matrix(A)
convert(::Type{Array{T}}, A::PartialQR) where {T} = convert(Array{T}, Matrix(A))
convert(::Type{Matrix}, A::PartialQR) = Matrix(A)
convert(::Type{Matrix{T}}, A::PartialQR) where {T} = convert(Array{T}, Matrix(A))
Array(A::PartialQR)  = convert(Array, A)

copy(A::PartialQR) = PartialQR(copy(A.Q), copy(A.R), copy(A.p))

  Matrix(A::PartialQR) = rmul!(A[:Q]*A[:R], A[:P]')
  adjoint(A::PartialQR) = Adjoint(A)
  transpose(A::PartialQR) = Transpose(A)






function getindex(A::PartialQR, d::Symbol)
  if     d == :P  return ColumnPermutation(A.p)
  elseif d == :Q  return A.Q
  elseif d == :R  return UpperTrapezoidal(A.R)
  elseif d == :k  return size(A.Q, 2)
  elseif d == :p  return A.p
  else            throw(KeyError(d))
  end
end

ishermitian(::PartialQR) = false
issym(::PartialQR) = false

isreal(::PartialQR{T}) where {T} = T <: Real

ndims(::PartialQR) = 2

size(A::PartialQR) = (size(A.Q,1), size(A.R,2))
size(A::PartialQR, dim::Integer) =
  dim == 1 ? size(A.Q,1) : (dim == 2 ? size(A.R,2) : 1)

# BLAS/LAPACK multiplication/division routines

  ## left-multiplication

function mul!!(C::AbstractVector{T}, A::PartialQR{T}, B::AbstractVector{T}) where T
    lmul!(A[:P]', B)
    mul!(C, A[:Q], A[:R]*B)
end  # overwrites B
function mul!!(C::AbstractMatrix{T}, A::PartialQR{T}, B::AbstractMatrix{T}) where T
    lmul!(A[:P]', B)
    mul!(C, A[:Q], A[:R]*B)
end  # overwrites B
mul!(C::AbstractVector{T}, A::PartialQR{T}, B::AbstractVector{T}) where {T} =
    mul!!(C, A, copy(B))

mul!(C::AbstractMatrix{T}, A::PartialQR{T}, B::AbstractMatrix{T}) where {T} =
    mul!!(C, A, copy(B))

  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function mul!(C::AbstractMatrix{T}, A::PartialQR{T}, Bc::$Adj{T,<:AbstractMatrix{T}}) where T
        tmp = $Adj(A[:P]) * Bc
        mul!(C, A[:Q], A[:R]*tmp)
      end
      function mul!(C::AbstractVector{T}, Ac::$Adj{T,PartialQR{T}}, B::AbstractVector{T}) where T
        A = parent(Ac)
        tmp = $Adj(A[:Q]) * B
        mul!(C, $Adj(A[:R]), tmp)
        lmul!(A[:P], C)
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,PartialQR{T}}, B::AbstractMatrix{T}) where T
        A = parent(Ac)
        tmp = $Adj(A[:Q]) * B
        mul!(C, $Adj(A[:R]), tmp)
        lmul!(A[:P], C)
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,PartialQR{T}}, Bc::$Adj{T,<:AbstractMatrix{T}}) where T
        A = parent(Ac)
        tmp = $Adj(A[:Q]) * Bc
        mul!(C, $Adj(A[:R]), tmp)
        lmul!(A[:P], C)
      end
    end
  end


  ## right-multiplication

  function mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::PartialQR{T}) where T
    mul!(C, A*B[:Q], B[:R])
    rmul!(C, B[:P]')
  end

  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function mul!!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bc::$Adj{T,PartialQR{T}}) where T
        B = parent(Bc)
        rmul!(A, B[:P])
        tmp = A * $Adj(B[:R])
        mul!(C, tmp, $Adj(B[:Q]))
      end  # overwrites A
      mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bc::$Adj{T,PartialQR{T}}) where {T} =
        mul!!(C, copy(A), Bc)
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, B::PartialQR{T}) where T
        tmp = Ac * B[:Q]
        mul!(C, tmp, B[:R])
        rmul!(C, $Adj(B[:P]))
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, Bc::$Adj{T,PartialQR{T}}) where T
        B = parent(Bc)
        tmp = Ac * B[:P]
        tmp = tmp * $Adj(B[:R])
        mul!(C, tmp, $Adj(B[:Q]))
      end
    end
  end


  ## left-division (pseudoinverse left-multiplication)
  function ldiv!(C::AbstractVector{T}, A::PartialQR{T}, B::AbstractVector{T}) where T
    tmp = (A[:R]*A.R')\(A[:Q]'*B)
    mul!(C, A[:R]', tmp)
    lmul!(A[:P], C)
  end

function ldiv!(C::AbstractMatrix{T}, A::PartialQR{T}, B::AbstractMatrix{T}) where T
    tmp = (A[:R]*A.R')\(A[:Q]'*B)
    mul!(C, A[:R]', tmp)
    lmul!(A[:P], C)
end

  # standard operations

  ## left-multiplication

  function *(A::PartialQR{TA}, B::AbstractVector{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(PartialQR{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1))
    mul!(CT, AT, BT)
  end
  function *(A::PartialQR{TA}, B::AbstractMatrix{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(PartialQR{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end
  function *(A::AbstractMatrix{TA}, B::PartialQR{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = (T == TA ? A : convert(Array{T}, A))
    BT = convert(PartialQR{T}, B)
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end

  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function *(Ac::$Adj{TA,PartialQR{TA}}, B::AbstractVector{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = convert(PartialQR{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1))
        mul!(CT, $Adj(AT), BT)
      end
      function *(A::PartialQR{TA}, Bc::$Adj{TB,<:AbstractMatrix{TB}}) where {TA,TB}
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = convert(PartialQR{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,1), size(Bc,2))
        mul!(CT, AT, $Adj(BT))
      end
      function *(Ac::$Adj{TA,PartialQR{TA}}, B::AbstractMatrix{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = convert(PartialQR{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(Ac::$Adj{TA,PartialQR{TA}}, Bc::$Adj{TB,<:AbstractMatrix{TB}}) where {TA,TB}
        A = parent(Ac)
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = convert(PartialQR{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1), size(Bc,2))
        mul!(CT, $Adj(AT), $Adj(BT))
      end
      function *(A::AbstractMatrix{TA}, Bc::$Adj{TB,PartialQR{TB}}) where {TA,TB}
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(PartialQR{T}, B)
        CT = Array{T}(undef, size(A,1), size(Bc,2))
        mul!(CT, AT, $Adj(BT))
      end
      function *(Ac::$Adj{TA,<:AbstractMatrix{TA}}, B::PartialQR{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(PartialQR{T}, B)
        CT = Array{T}(undef, size(Ac,1), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(Ac::$Adj{TA,<:AbstractMatrix{TA}}, Bc::$Adj{TB,PartialQR{TB}}) where {TA,TB}
        A = parent(Ac)
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(PartialQR{T}, B)
        CT = Array{T}(undef, size(Ac,1), size(Bc,2))
        mul!(CT, $Adj(AT), $Adj(BT))
      end
    end
  end


## left-division
function \(A::PartialQR{TA}, B::AbstractVector{TB}) where {TA,TB}
  T = promote_type(TA, TB)
  AT = convert(PartialQR{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(undef, size(A,2))
  ldiv!(CT, AT, BT)
end
function \(A::PartialQR{TA}, B::AbstractMatrix{TB}) where {TA,TB}
  T = promote_type(TA, TB)
  AT = convert(PartialQR{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(undef, size(A,2), size(B,2))
  ldiv!(CT, AT, BT)
end

# factorization routines

  for sfx in ("", "!")
    f = Symbol("pqrfact", sfx)
    g = Symbol("pqrfact_none", sfx)
    h = Symbol("pqr", sfx)
    @eval begin
      function $f(
          trans::Symbol, A::AbstractMatOrLinOp{S}, opts::LRAOptions=LRAOptions(S);
          args...) where S
        chktrans(trans)
        opts = copy(opts; args...)
        chkopts!(opts, A)
        opts.sketch == :none && return $g(trans, A, opts)
        V = idfact(trans, A, opts)
        F = qr!(convert(Matrix, getcols(trans, A, V[:sk])))
        retq = occursin("q", opts.pqrfact_retval)
        retr = occursin("r", opts.pqrfact_retval)
        rett = occursin("t", opts.pqrfact_retval)
        Q = retq ? Matrix(F.Q) : nothing
        R = retr ? pqrr(F.R, V[:T]) : nothing
        T = rett ? V[:T] : nothing
        retq && retr && !rett && return PartialQR(Q, R, V[:p])
        PQRFactors(Q, R, V[:p], V[:k], T)
      end
      $f(trans::Symbol, A, args...; kwargs...) =
        $f(trans, LinOp(A), args...; kwargs...)
      $f(A, args...; kwargs...) = $f(:n, A, args...; kwargs...)

      function $h(trans::Symbol, A, args...; kwargs...)
        push!(kwargs, (:pqrfact_retval, "qr"))
        F = $f(trans, A, args...; kwargs...)
        F.Q, F.R, F.p
      end
      $h(A, args...; kwargs...) = $h(:n, A, args...; kwargs...)
    end
  end


pqrfact_none!(trans::Symbol, A::AbstractMatrix, opts::LRAOptions) =
  pqrfact_backend!(convert(Matrix, trans == :n ? A : A'), opts)

  pqrfact_none(trans::Symbol, A::AbstractMatrix, opts::LRAOptions) =
    pqrfact_backend!(Matrix(trans == :n ? A : A'), opts)


function pqrr(R::Matrix{S}, T::Matrix{S}) where S
  k, n = size(T)
  n += k
  R_ = Array{S}(undef, k, n)
  R1 = view(R_, :,   1:k)
  R2 = view(R_, :, k+1:n)
  copyto!(R1, R)
  copyto!(R2, T)
  lmul!(UpperTriangular(R1), R2)
  R_
end

## core backend routine: rank-adaptive GEQP3 with determinant maximization
function pqrfact_backend!(A::AbstractMatrix, opts::LRAOptions)
  p, tau, k = geqp3_adap!(A, opts)
  pqrback_postproc(A, p, tau, k, opts)
end

function geqp3_adap!(A::AbstractMatrix{T}, opts::LRAOptions) where T
  m, n = size(A)
  jpvt = collect(BlasInt, 1:n)
  l    = min(m, n)
  k    = (opts.rank < 0 || opts.rank > l) ? l : opts.rank
  tau  = Array{T}(undef, k)
  if k > 0
    k = geqp3_adap_main!(A, jpvt, tau, opts)
  end
  jpvt = convert(Array{Int}, jpvt)
  jpvt, tau, k
end

function geqp3_adap_main!(
    A::AbstractMatrix{T}, jpvt::Vector{BlasInt}, tau::Vector{T},
    opts::LRAOptions) where T<:BlasFloat
  chkstride1(A)
  lda = stride(A, 2)
  n   = length(jpvt)
  k   = length(tau)

  # set block size and allocate work array
  nb      = min(opts.nb, k)
  is_real = T <: Real
  lwork   = 2*n*is_real + (n + 1)*nb
  work    = Array{T}(undef, lwork)

  # initialize column norms
  if is_real
    @inbounds for j = 1:n
      work[j] = work[n+j] = norm(view(A,:,j))
    end
  else
    rwork = Array{eltype(real(zero(T)))}(undef, 2*n)
    @inbounds for j = 1:n
      rwork[j] = rwork[n+j] = norm(view(A,:,j))
    end
  end
  maxnrm = maximum(view(is_real ? work : rwork, 1:n))

  # set pivot threshold
  ptol = max(opts.atol, opts.rtol*maxnrm)

  # block factorization
  j = 1
  fjb = Ref{BlasInt}()
  while j <= k
    jb = BlasInt(min(nb, k-j+1))
    if is_real
      _LAPACK.laqps!(
        BlasInt(j-1), jb, fjb, view(A,:,j:n), view(jpvt,j:n), view(tau,j:k),
        view(work,j:n), view(work,n+j:2*n),
        view(work,2*n+1:2*n+nb), view(work,2*n+jb+1:lwork))
    else
      _LAPACK.laqps!(
        BlasInt(j-1), jb, fjb, view(A,:,j:n), view(jpvt,j:n), view(tau,j:k),
        view(rwork,j:n), view(rwork,n+j:2*n),
        view(work,1:nb), view(work,jb+1:lwork))
    end
    jn = j + fjb[]

    # check for rank termination
    if abs(A[jn-1,jn-1]) <= ptol
      @inbounds for i = j:jn-1
        abs(A[i,i]) <= ptol && return i - 1
      end
    end
    j = jn
  end
  k
end

function pqrback_postproc(
    A::AbstractMatrix{S}, p::Vector{Int}, tau::Vector{S}, k::Integer,
    opts::LRAOptions) where S
  retq = occursin("q", opts.pqrfact_retval)
  retr = occursin("r", opts.pqrfact_retval)
  rett = occursin("t", opts.pqrfact_retval)
  maxdet = 0 < k < size(A,2) && opts.maxdet_tol >= 0
  Q = retq ? LAPACK.orgqr!(A[:,1:k], tau, k) : nothing
  R = retr || rett || maxdet ? triu!(A[1:k,:]) : nothing
  T = rett || maxdet ? maxdet_t(R) : nothing
  if maxdet
    maxdet_swapcols!(Q, R, p, T, opts)
    R = retr ? R : nothing
  end
  retq && retr && !rett && return PartialQR(Q, R, p)
  PQRFactors(Q, R, p, k, T)
end

function maxdet_t(R::AbstractMatrix{S}) where S
  k, n = size(R)
  T = R[:,k+1:n]
  ldiv!(UpperTriangular(view(R,1:k,1:k)), T)
end

## rank-revealing QR determinant maximization
function maxdet_swapcols!(
    Q::Union{Matrix{S}, Nothing}, R::Matrix{S}, p::Vector{Int}, T::Matrix{S},
    opts::LRAOptions) where S
  k, n  = size(R)
  R1    = view(R, :, 1:k)
  work  = Array{S}(undef, max(n, 2*k))
  retq  = occursin("q", opts.pqrfact_retval)
  retr  = occursin("r", opts.pqrfact_retval)
  niter = 0
  while true
    Tmax, idx = findmaxabs(T)
    Tmax <= 1 + opts.maxdet_tol && break
    if niter == opts.maxdet_niter
      opts.verb &&
        warn("iteration limit ($niter) reached in determinant maximization")
      break
    end
    niter += 1
    i, j = Tuple(ind2sub((k, n-k), idx))
    maxdet_update!(R1, p, T, work, i, j, retr)
  end
  niter == 0 && return
  F = qrfact!(R1)
  retq && LAPACK.gemqrt!('R', 'N', F.factors, F.T, Q)
  if retr
    triu!(R1)
    R2 = view(R, :, k+1:n)
    copyto!(R2, T)
    lmul!(UpperTriangular(R1), R2)
  end
end

## column swap update based on Sherman-Morrison
function maxdet_update!(
    R1::AbstractMatrix{S}, p::Vector{Int}, T::Matrix{S}, work::Vector{S},
    i::Integer, j::Integer, retr::Bool) where S
  k, n = size(T)
  n += k
  p[i], p[k+j] = p[k+j], p[i]
  @inbounds @simd for l = 1:k
    work[l] = T[l,j]
    T[l,j]  = 0
  end
  work[i] -= 1
  T[i,j]   = 1
  @inbounds for l = 1:n-k
    work[k+l] = conj(T[i,l])
  end
  BLAS.ger!(-1/(1 + work[i]), view(work,1:k), view(work,k+1:n), T)
  if retr
    mul!(view(work,k+1:2*k), R1, view(work,1:k))
    @inbounds @simd for l = 1:k
      R1[l,i] += work[k+l]
    end
  end
end

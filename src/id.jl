#= id.jl

References:

  H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. On the compression of low
    rank matrices. SIAM J. Sci. Comput. 26 (4): 1389-1404, 2005.

  E. Liberty, F. Woolfe, P.-G. Martinsson V. Rokhlin, M. Tygert. Randomized
    algorithms for the low-rank approximation of matrices. Proc. Natl. Acad.
    Sci. U.S.A. 104 (51): 20167-20172, 2007.
=#

# IDPackedV

type IDPackedV{S} <: Factorization{S}
  sk::Vector{Int}
  rd::Vector{Int}
  T::Matrix{S}
end

conj!(A::IDPackedV) = IDPackedV(A.sk, A.rd, conj!(A.T))
conj(A::IDPackedV) = IDPackedV(A.sk, A.rd, conj(A.T))

convert{T}(::Type{IDPackedV{T}}, A::IDPackedV) =
  IDPackedV(A.sk, A.rd, convert(Array{T}, A.T))
convert(::Type{Array}, A::IDPackedV) = full(A)
convert{T}(::Type{Array{T}}, A::IDPackedV) = convert(Array{T}, full(A))

copy(A::IDPackedV) = IDPackedV(copy(A.sk), copy(A.rd), copy(A.T))

full(A::IDPackedV) = A_mul_Bc!([eye(A[:k]) A[:T]], A[:P])

function getindex(A::IDPackedV, d::Symbol)
  if     d == :P   return ColumnPermutation([A.sk; A.rd])
  elseif d == :T   return A.T
  elseif d == :k   return length(A.sk)
  elseif d == :rd  return A.rd
  elseif d == :sk  return A.sk
  else             throw(KeyError(d))
  end
end

ishermitian(A::IDPackedV) = false
issym(A::IDPackedV) = false

isreal{T}(A::IDPackedV{T}) = T <: Real

ndims(A::IDPackedV) = 2

size(A::IDPackedV) = (size(A.T,1), sum(size(A.T)))
size(A::IDPackedV, dim::Integer) =
  (dim == 1 ? size(A.T,1) : (dim == 2 ? sum(size(A.T)) : 1))

## BLAS/LAPACK multiplication routines

### left-multiplication

function A_mul_B!!{T}(
    C::StridedVecOrMat{T}, A::IDPackedV{T}, B::StridedVecOrMat{T})
  k, n = size(A)
  Ac_mul_B!(A[:P], B)
  copy!(C, sub(B,1:k,:))
  BLAS.gemm!('N', 'N', one(T), A[:T], sub(B,k+1:n,:), one(T), C)
end  # overwrites B
A_mul_B!{T}(C::StridedVecOrMat{T}, A::IDPackedV{T}, B::StridedVecOrMat{T}) =
  A_mul_B!!(C, A, copy(B))

for (f!, g) in ((:A_mul_Bc!, :Ac_mul_Bc), (:A_mul_Bt!, :At_mul_Bt))
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::IDPackedV{T}, B::StridedMatrix{T})
      k, n = size(A)
      tmp = $g(A[:P], B)
      copy!(C, sub(tmp,1:k,:))
      BLAS.gemm!('N', 'N', one(T), A[:T], sub(tmp,k+1:n,:), one(T), C)
    end
  end
end

for f in (:Ac_mul_B!, :At_mul_B!)
  @eval begin
    function $f{T}(
        C::StridedVecOrMat{T}, A::IDPackedV{T}, B::StridedVecOrMat{T})
      k, n = size(A)
      copy!(sub(C,1:k,:), B)
      $f(sub(C,k+1:n,:), A[:T], B)
      A_mul_B!(A[:P], C)
    end
  end
end

for (f, g) in ((:Ac_mul_Bc!, :ctranspose!), (:At_mul_Bt!, :transpose!))
  @eval begin
    function $f{T}(C::StridedMatrix{T}, A::IDPackedV{T}, B::StridedMatrix{T})
      k, n = size(A)
      $g(sub(C,1:k,:), B)
      $f(sub(C,k+1:n,:), A[:T], B)
      A_mul_B!(A[:P], C)
    end
  end
end

### right-multiplication

function A_mul_B!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T})
  k, n = size(B)
  copy!(sub(C,:,1:k), A)
  A_mul_B!(sub(C,:,k+1:n), A, B[:T])
  A_mul_Bc!(C, B[:P])
end

for (f!, trans) in ((:A_mul_Bc!, 'C'), (:A_mul_Bt!, 'T'))
  f!! = symbol(f!, "!")
  @eval begin
    function $f!!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T})
      k, n = size(B)
      A_mul_B!(A, B[:P])
      copy!(C, sub(A,:,1:k))
      BLAS.gemm!('N', $trans, one(T), sub(A,:,k+1:n), B[:T], one(T), C)
    end  # overwrites A
    $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T}) =
      $f!!(C, copy(A), B)
  end
end

for (f, g, h) in ((:Ac_mul_B!, :ctranspose!, :A_mul_Bc!),
                  (:At_mul_B!, :transpose!,  :A_mul_Bt!))
  @eval begin
    function $f{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T})
      k, n = size(B)
      $g(sub(C,:,1:k), A)
      $f(sub(C,:,k+1:n), A, B[:T])
      $h(C, B[:P])
    end
  end
end

for (f!, g, trans) in ((:Ac_mul_Bc!, :Ac_mul_B, 'C'),
                       (:At_mul_Bt!, :At_mul_B, 'T'))
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T})
      k, n = size(B)
      tmp = $g(A, B[:P])
      copy!(C, sub(tmp,:,1:k))
      BLAS.gemm!('N', $trans, one(T), sub(tmp,:,k+1:n), B[:T], one(T), C)
    end
  end
end

# ID

type ID{S} <: Factorization{S}
  C::Matrix{S}
  sk::Vector{Int}
  rd::Vector{Int}
  T::Matrix{S}
end

ID{T}(A::AbstractMatrix{T}, V::IDPackedV{T}) = ID(A[:,V.sk], V.sk, V.rd, V.T)
function ID{T}(A::AbstractLinOp{T}, V::IDPackedV{T})
  k = V[:k]
  S = zeros(size(A,2), k)
  for i = 1:k
    S[V.sk[i],i] = 1
  end
  ID(A*S, V.sk, V.rd, V.T)
end

conj!(A::ID) = ID(conj!(A.C), A.sk, A.rd, conj!(A.T))
conj(A::ID) = conj!(copy(A))

convert{T}(::Type{ID{T}}, A::ID) =
  ID(convert(Array{T}, A.C), A.sk, A.rd, convert(Array{T}, A.T))
convert(::Type{Array}, A::ID) = full(A)
convert{T}(::Type{Array{T}}, A::ID) = convert(Array{T}, full(A))

copy(A::ID) = ID(copy(A.C), copy(A.sk), copy(A.rd), copy(A.T))

full(A::ID) = A[:C]*A[:V]

function getindex{T}(A::ID{T}, d::Symbol)
  if     d == :C   return A.C
  elseif d == :P   return ColumnPermutation([A.sk; A.rd])
  elseif d == :T   return A.T
  elseif d == :V   return IDPackedV(A.sk, A.rd, A.T)
  elseif d == :k   return length(A.sk)
  elseif d == :rd  return A.rd
  elseif d == :sk  return A.sk
  else             throw(KeyError(d))
  end
end

ishermitian(A::ID) = false
issym(A::ID) = false

isreal{T}(A::ID{T}) = T <: Real

ndims(A::ID) = 2

size(A::ID) = (size(A.C,1), sum(size(A.T)))
size(A::ID, dim::Integer) =
  (dim == 1 ? size(A.C,1) : (dim == 2 ? sum(size(A.T)) : 1))

## BLAS/LAPACK multiplication routines

### left-multiplication

function A_mul_B!!{T}(y::StridedVector{T}, A::ID{T}, x::StridedVector{T})
  tmp = Array(T, A[:k])
  A_mul_B!!(tmp, A[:V], x)
  A_mul_B!(y, A[:C], tmp)
end  # overwrites x
function A_mul_B!!{T}(C::StridedMatrix{T}, A::ID{T}, B::StridedMatrix{T})
  tmp = Array(T, A[:k], size(B,2))
  A_mul_B!!(tmp, A[:V], B)
  A_mul_B!(C, A[:C], tmp)
end  # overwrites B
A_mul_B!{T}(C::StridedVecOrMat{T}, A::ID{T}, B::StridedVecOrMat{T}) =
  A_mul_B!!(C, A, copy(B))

for f in (:A_mul_Bc, :A_mul_Bt)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::ID{T}, B::StridedMatrix{T})
      tmp = $f(A[:V], B)
      A_mul_B!(C, A[:C], tmp)
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedVecOrMat{T}, A::ID{T}, B::StridedVecOrMat{T})
      tmp = $f(A[:C], B)
      $f!(C, A[:V], tmp)
    end
  end
end

for (f, g!) in ((:Ac_mul_Bc, :Ac_mul_B!), (:At_mul_Bt, :At_mul_B!))
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::ID{T}, B::StridedMatrix{T})
      tmp = $f(A[:C], B)
      $g!(C, A[:V], tmp)
    end
  end
end

### right-multiplication

A_mul_B!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) =
  A_mul_B!(C, A*B[:C], B[:V])

for f! in (:A_mul_Bc!, :A_mul_Bt!)
  f!! = symbol(f!, "!")
  @eval begin
    function $f!!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T})
      tmp = Array(T, size(A,1), B[:k])
      $f!!(tmp, A, B[:V])
      $f!(C, tmp, B[:C])
    end  # overwrites A
    $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) =
      $f!!(C, copy(A), B)
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T})
      tmp = $f(A, B[:C])
      A_mul_B!(C, tmp, B[:V])
    end
  end
end

for (f, g!) in ((:Ac_mul_Bc, :A_mul_Bc!), (:At_mul_Bt, :A_mul_Bt!))
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T})
      tmp = $f(A, B[:V])
      $g!(C, tmp, B[:C])
    end
  end
end

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  for t in (:IDPackedV, :ID)
    @eval begin
      function $f{TA,TB}(A::$t{TA}, B::StridedVector{TB})
        T = promote_type(TA, TB)
        AT = convert($t{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array(T, size(A,$i))
        $f!(CT, AT, BT)
      end
    end
  end
end

for (f, f!, i, j) in ((:*,         :A_mul_B!,   1, 2),
                      (:A_mul_Bc,  :A_mul_Bc!,  1, 1),
                      (:A_mul_Bt,  :A_mul_Bt!,  1, 1),
                      (:Ac_mul_B,  :Ac_mul_B!,  2, 2),
                      (:Ac_mul_Bc, :Ac_mul_Bc!, 2, 1),
                      (:At_mul_B,  :At_mul_B!,  2, 2),
                      (:At_mul_Bt, :At_mul_Bt!, 2, 1))
  for t in (:IDPackedV, :ID)
    @eval begin
      function $f{TA,TB}(A::$t{TA}, B::StridedMatrix{TB})
        T = promote_type(TA, TB)
        AT = convert($t{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array(T, size(A,$i), size(B,$j))
        $f!(CT, AT, BT)
      end
    end
  end
end

## right-multiplication
for (f, f!, i, j) in ((:*,         :A_mul_B!,   1, 2),
                      (:A_mul_Bc,  :A_mul_Bc!,  1, 1),
                      (:A_mul_Bt,  :A_mul_Bt!,  1, 1),
                      (:Ac_mul_B,  :Ac_mul_B!,  2, 2),
                      (:Ac_mul_Bc, :Ac_mul_Bc!, 2, 1),
                      (:At_mul_B,  :At_mul_B!,  2, 2),
                      (:At_mul_Bt, :At_mul_Bt!, 2, 1))
  for t in (:IDPackedV, :ID)
    @eval begin
      function $f{TA,TB}(A::StridedMatrix{TA}, B::$t{TB})
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert($t{T}, B)
        CT = Array(T, size(A,$i), size(B,$j))
        $f!(CT, AT, BT)
      end
    end
  end
end

# factorization routines

for sfx in ("", "!")
  f = symbol("idfact", sfx)
  g = symbol("id", sfx)
  h = symbol("pqrfact", sfx)
  @eval begin
    function $f{S}(A::AbstractMatOrLinOp{S}, opts::LRAOptions)
      chkopts(opts)
      if opts.sketch in (:none, :subs)  F = $h(A, opts)
      else                              F = sketchfact(:left, :n, A, opts)
      end
      k, n = size(F.R)
      sk = F.p[1:k]
      rd = F.p[k+1:n]
      T = F.R[1:k,k+1:n]
      BLAS.trsm!('L', 'U', 'N', 'N', one(S), sub(F.R,1:k,1:k), T)
      IDPackedV(sk, rd, T)
    end
    function $f(A::AbstractMatOrLinOp, rank_or_rtol::Real)
      opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol)
                               : LRAOptions(rank=rank_or_rtol))
      $f(A, opts)
    end
    $f{T}(A::AbstractMatOrLinOp{T}) = $f(A, default_rtol(T))
    $f(A, args...) = $f(LinOp(A), args...)

    function $g(A, args...)
      V = $f(A, args...)
      V.sk, V.rd, V.T
    end
  end
end
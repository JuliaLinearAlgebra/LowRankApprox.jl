#= src/lapack.jl
=#

module _LAPACK

import Base.LinAlg: BlasFloat, BlasInt, chkstride1
import Base.blasfunc

const liblapack = Base.liblapack_name

for (lapmr, lapmt, elty) in ((:slapmr_, :slapmt_, :Float32   ),
                             (:dlapmr_, :dlapmt_, :Float64   ),
                             (:clapmr_, :clapmt_, :Complex64 ),
                             (:zlapmr_, :zlapmt_, :Complex128))
  @eval begin
    function lapmr!(
        forward::BlasInt, X::StridedVecOrMat{$elty}, K::StridedVector{BlasInt})
      chkstride1(X, K)
      m = size(X, 1)
      n = size(X, 2)
      k = length(K)
      m == k || throw(DimensionMismatch)
      ldx = max(1, stride(X,2))
      ccall(($(blasfunc(lapmr)), liblapack), Void,
            (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
             Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
            &forward, &m, &n,
            X, &ldx, K)
      X
    end

    function lapmt!(
        forward::BlasInt, X::StridedMatrix{$elty}, K::StridedVector{BlasInt})
      chkstride1(X, K)
      m, n = size(X)
      k = length(K)
      n == k || throw(DimensionMismatch)
      ldx = max(1, stride(X,2))
      ccall(($(blasfunc(lapmt)), liblapack), Void,
            (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
             Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
            &forward, &m, &n,
            X, &ldx, K)
      X
    end
  end
end

for (laqps, elty, relty) in ((:slaqps_, :Float32,    :Float32),
                             (:dlaqps_, :Float64,    :Float64),
                             (:claqps_, :Complex64,  :Float32),
                             (:zlaqps_, :Complex128, :Float64))
  @eval begin
    function laqps!(
        offset::BlasInt, nb::BlasInt, kb::StridedVector{BlasInt},
        A::StridedMatrix{$elty},
        jpvt::StridedVector{BlasInt}, tau::StridedVector{$elty},
        vn1::StridedVector{$relty}, vn2::StridedVector{$relty},
        auxv::StridedVector{$elty}, F::StridedVector{$elty})
      m, n = size(A)
      lda  = max(1, stride(A,2))
      ldf  = max(1, n)
      ccall(
        ($(blasfunc(laqps)), liblapack), Void,
        (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
         Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
         Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
        &m, &n, &offset, &nb, kb,
        A, &lda, jpvt, tau,
        vn1, vn2, auxv, F, &ldf)
    end
  end
end

end  # module
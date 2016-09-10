#= src/lapack.jl
=#

module _LAPACK

import Base.LinAlg: BlasFloat, BlasInt, chkstride1
import Base.blasfunc

const liblapack = Base.liblapack_name

for (laqps, elty, relty) in ((:slaqps_, :Float32,    :Float32),
                             (:dlaqps_, :Float64,    :Float64),
                             (:claqps_, :Complex64,  :Float32),
                             (:zlaqps_, :Complex128, :Float64))
  @eval begin
    function laqps!(
        offset::BlasInt, nb::BlasInt, kb::Ref{BlasInt},
        A::StridedMatrix{$elty},
        jpvt::StridedVector{BlasInt}, tau::StridedVector{$elty},
        vn1::StridedVector{$relty}, vn2::StridedVector{$relty},
        auxv::StridedVector{$elty}, F::StridedVector{$elty})
      m, n = size(A)
      ccall(
        ($(blasfunc(laqps)), liblapack), Void,
        (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
         Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
         Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
        &m, &n, &offset, &nb, kb,
        A, &max(1,stride(A,2)), jpvt, tau,
        vn1, vn2, auxv, F, &max(1,n))
    end
  end
end

end  # module
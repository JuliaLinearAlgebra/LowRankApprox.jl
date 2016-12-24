#= src/lapack.jl
=#

module _LAPACK

using Base.BLAS.@blasfunc
using Base.LinAlg: BlasFloat, BlasInt, chkstride1

const liblapack = Base.liblapack_name

for (geqrf, gelqf, orgqr, orglq, elty) in
      ((:sgeqrf_, :sgelqf_, :sorgqr_, :sorglq_, :Float32   ),
       (:dgeqrf_, :dgelqf_, :dorgqr_, :dorglq_, :Float64   ),
       (:cgeqrf_, :cgelqf_, :cungqr_, :cunglq_, :Complex64 ),
       (:zgeqrf_, :zgelqf_, :zungqr_, :zunglq_, :Complex128))
  @eval begin
    function geqrf!(
        A::StridedMatrix{$elty}, tau::Vector{$elty}, work::Vector{$elty})
      chkstride1(A)
      m, n  = size(A)
      k     = min(m, n)
      tau   = length(tau) < k ? Array($elty, k) : tau
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
        ccall((@blasfunc($geqrf), liblapack), Void,
              (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
               Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
              &m, &n, A, &max(1,stride(A,2)),
              tau, work, &lwork, info)
        if i == 1
          lwork = BlasInt(real(work[1]))
          work  = length(work) < lwork ? Array($elty, lwork) : work
          lwork = length(work)
        end
      end
      A, tau, work
    end

    function gelqf!(
        A::StridedMatrix{$elty}, tau::Vector{$elty}, work::Vector{$elty})
      chkstride1(A)
      m, n  = size(A)
      k     = min(m, n)
      tau   = length(tau) < k ? Array($elty, k) : tau
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
        ccall((@blasfunc($gelqf), liblapack), Void,
              (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
               Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
              &m, &n, A, &max(1,stride(A,2)),
              tau, work, &lwork, info)
        if i == 1
            lwork = BlasInt(real(work[1]))
            work  = length(work) < lwork ? Array($elty, lwork) : work
            lwork = length(work)
        end
      end
      A, tau, work
    end

    function orglq!(
        A::StridedMatrix{$elty}, tau::Vector{$elty}, k::Integer,
        work::Vector{$elty})
      chkstride1(A)
      n = size(A, 2)
      m = min(n, size(A, 1))
      0 <= k <= min(m, length(tau)) || throw(DimensionMismatch)
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
          ccall((@blasfunc($orglq), liblapack), Void,
                (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{BlasInt}),
                &m, &n, &k, A,
                &max(1,stride(A,2)), tau, work, &lwork,
                info)
          if i == 1
              lwork = BlasInt(real(work[1]))
              work  = length(work) < lwork ? Array($elty, lwork) : work
              lwork = length(work)
          end
      end
      A, tau, work
    end

    function orgqr!(
        A::StridedMatrix{$elty}, tau::Vector{$elty}, k::Integer,
        work::Vector{$elty})
      chkstride1(A)
      m = size(A, 1)
      n = min(m, size(A,2))
      0 <= k <= min(n, length(tau)) || throw(DimensionMismatch)
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
        ccall((@blasfunc($orgqr), liblapack), Void,
              (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
               Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
               Ptr{BlasInt}),
              &m, &n, &k, A,
              &max(1,stride(A,2)), tau, work, &lwork,
              info)
        if i == 1
          lwork = BlasInt(real(work[1]))
          work  = length(work) < lwork ? Array($elty, lwork) : work
          lwork = length(work)
        end
      end
      A, tau, work
    end
  end
end

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
        (@blasfunc($laqps), liblapack), Void,
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
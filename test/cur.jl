#= test/cur.jl
=#

println("cur.jl")
tic()

m = 128
n =  64
M = matrixlib(:fourier, rand(m), rand(n))
Mh = M[1:n,1:n]; Mh += Mh';
Ms = M[1:n,1:n]; Ms += Ms.';
opts = LRAOptions(maxdet_tol=0., sketch_randn_niter=1)

for (t, s) in ((:none,                 :none ),
               (:RandomGaussian,       :randn),
               (:RandomSubset,         :sub  ),
               (:SRFT,                 :srft ),
               (:SparseRandomGaussian, :sprn ))
  opts.sketch = s
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    rtol = 5*eps(real(T))
    approx_rtol = 1000*rtol
    opts.rtol = rtol

    for (N, p, q) in ((M, m, n), (Mh, n, n), (Ms, n, n))
      A = convert(Array{T}, T <: Real ? real(N) : N)

      U  = curfact(A, opts)
      F  = CUR(A, U)
      @test norm(A - full(F)) < approx_rtol*norm(A)

      xp = rand(T, p)
      xq = rand(T, q)
      y = A  *xq; @test norm(y - F  *xq) < approx_rtol*norm(y)
      y = A' *xp; @test norm(y - F' *xp) < approx_rtol*norm(y)
      y = A.'*xp; @test norm(y - F.'*xp) < approx_rtol*norm(y)

      Bp = rand(T, p, p)
      Bq = rand(T, q, q)
      C = A   *Bq  ; @test norm(C - F   *Bq  ) < approx_rtol*norm(C)
      C = A   *Bq' ; @test norm(C - F   *Bq' ) < approx_rtol*norm(C)
      C = A   *Bq.'; @test norm(C - F   *Bq.') < approx_rtol*norm(C)
      C = A'  *Bp  ; @test norm(C - F'  *Bp  ) < approx_rtol*norm(C)
      C = A'  *Bp' ; @test norm(C - F'  *Bp' ) < approx_rtol*norm(C)
      C = A.' *Bp  ; @test norm(C - F.' *Bp  ) < approx_rtol*norm(C)
      C = A.' *Bp.'; @test norm(C - F.' *Bp.') < approx_rtol*norm(C)
      C = Bp  *A   ; @test norm(C - Bp  *F   ) < approx_rtol*norm(C)
      C = Bq  *A'  ; @test norm(C - Bq  *F'  ) < approx_rtol*norm(C)
      C = Bq  *A.' ; @test norm(C - Bq  *F.' ) < approx_rtol*norm(C)
      C = Bp' *A   ; @test norm(C - Bp' *F   ) < approx_rtol*norm(C)
      C = Bq' *A'  ; @test norm(C - Bq' *F'  ) < approx_rtol*norm(C)
      C = Bp.'*A   ; @test norm(C - Bp.'*F   ) < approx_rtol*norm(C)
      C = Bq.'*A.' ; @test norm(C - Bq.'*F.' ) < approx_rtol*norm(C)
    end
  end
end

toc()
#= test/id.jl
=#

println("id.jl")
tic()

m = 128
n =  64
M = matrixlib(:fourier, rand(m), rand(n))
opts = LRAOptions(sketch_randn_niter=1)

for (t, s) in ((:none,                 :none ),
               (:RandomGaussian,       :randn),
               (:RandomSubset,         :sub  ),
               (:SRFT,                 :srft ),
               (:SparseRandomGaussian, :sprn ))
  opts.sketch = s
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    rtol = 5*eps(real(T))
    approx_rtol = 100*rtol
    opts.rtol = rtol

    A = convert(Array{T}, T <: Real ? real(M) : M)
    nrm = norm(A)

    V  = idfact(:n, A, opts)
    F  = ID(:n, A, V)
    Vc = idfact(:c, A, opts)
    Fc = ID(:c, A, Vc)
    @test norm(A  - full(F )) < approx_rtol*nrm
    @test norm(A' - full(Fc)) < approx_rtol*nrm

    for (N, G, p, q) in ((full(V), V, V[:k], n), (A, F, m, n))
      xp = rand(T, p)
      xq = rand(T, q)
      y = N  *xq; @test norm(y - G  *xq) < approx_rtol*norm(y)
      y = N' *xp; @test norm(y - G' *xp) < approx_rtol*norm(y)
      y = N.'*xp; @test norm(y - G.'*xp) < approx_rtol*norm(y)

      Bp = rand(T, p, p)
      Bq = rand(T, q, q)
      C = N   *Bq  ; @test norm(C - G   *Bq  ) < approx_rtol*norm(C)
      C = N   *Bq' ; @test norm(C - G   *Bq' ) < approx_rtol*norm(C)
      C = N   *Bq.'; @test norm(C - G   *Bq.') < approx_rtol*norm(C)
      C = N'  *Bp  ; @test norm(C - G'  *Bp  ) < approx_rtol*norm(C)
      C = N'  *Bp' ; @test norm(C - G'  *Bp' ) < approx_rtol*norm(C)
      C = N.' *Bp  ; @test norm(C - G.' *Bp  ) < approx_rtol*norm(C)
      C = N.' *Bp.'; @test norm(C - G.' *Bp.') < approx_rtol*norm(C)
      C = Bp  *N   ; @test norm(C - Bp  *G   ) < approx_rtol*norm(C)
      C = Bq  *N'  ; @test norm(C - Bq  *G'  ) < approx_rtol*norm(C)
      C = Bq  *N.' ; @test norm(C - Bq  *G.' ) < approx_rtol*norm(C)
      C = Bp' *N   ; @test norm(C - Bp' *G   ) < approx_rtol*norm(C)
      C = Bq' *N'  ; @test norm(C - Bq' *G'  ) < approx_rtol*norm(C)
      C = Bp.'*N   ; @test norm(C - Bp.'*G   ) < approx_rtol*norm(C)
      C = Bq.'*N.' ; @test norm(C - Bq.'*G.' ) < approx_rtol*norm(C)
    end
  end
end

toc()
#= test/pqr.jl
=#

  @testset "pqr" begin
    m = 128
    n =  64
    M = matrixlib(:fourier, rand(m), rand(n))
    opts = LRAOptions(maxdet_tol=0., sketch_randn_niter=1)

    @time for (t, s) in ((:none,                 :none ),
                         (:RandomGaussian,       :randn),
                         (:RandomSubset,         :sub  ),
                         (:SRFT,                 :srft ),
                         (:SparseRandomGaussian, :sprn ))
      opts.sketch = s
      for T in (Float32, Float64, ComplexF32, ComplexF64)
        let A
            println("  $t/$T")

            rtol = 5*eps(real(T))
            approx_rtol = 100*rtol
            opts.rtol = rtol

            A = convert(Array{T}, T <: Real ? real(M) : M)
            nrm = norm(A)

            F  = pqrfact(:n, A, opts)
            Fc = pqrfact(:c, A, opts)
            @test norm(A  - Matrix(F )) < approx_rtol*nrm
            @test norm(A' - Matrix(Fc)) < approx_rtol*nrm

            xm = rand(T, m)
            xn = rand(T, n)
            y = A  *xn; @test norm(y - F  *xn) < approx_rtol*norm(y)
            y = A' *xm; @test norm(y - F' *xm) < approx_rtol*norm(y)
            y = transpose(A)*xm; @test norm(y - transpose(F)*xm) < approx_rtol*norm(y)

            Bm = rand(T, m, m)
            Bn = rand(T, n, n)
            C = A   *Bn  ; @test norm(C - F   *Bn  ) < approx_rtol*norm(C)
            C = A   *Bn' ; @test norm(C - F   *Bn' ) < approx_rtol*norm(C)
            C = A * transpose(Bn); @test norm(C - F   *transpose(Bn)) < approx_rtol*norm(C)
            C = A'  *Bm  ; @test norm(C - F'  *Bm  ) < approx_rtol*norm(C)
            C = A'  *Bm' ; @test norm(C - F'  *Bm' ) < approx_rtol*norm(C)
            C = transpose(A) *Bm  ; @test norm(C - transpose(F) *Bm  ) < approx_rtol*norm(C)
            C = transpose(A) *transpose(Bm); @test norm(C - transpose(F) *transpose(Bm)) < approx_rtol*norm(C)
            C = Bm  *A   ; @test norm(C - Bm  *F   ) < approx_rtol*norm(C)
            C = Bn  *A'  ; @test norm(C - Bn  *F'  ) < approx_rtol*norm(C)
            C = Bn  *transpose(A) ; @test norm(C - Bn  *transpose(F) ) < approx_rtol*norm(C)
            C = Bm' *A   ; @test norm(C - Bm' *F   ) < approx_rtol*norm(C)
            C = Bn' *A'  ; @test norm(C - Bn' *F'  ) < approx_rtol*norm(C)
            C = transpose(Bm)*A   ; @test norm(C - transpose(Bm)*F   ) < approx_rtol*norm(C)
            C = transpose(Bn)*transpose(A) ; @test norm(C - transpose(Bn)*transpose(F) ) < approx_rtol*norm(C)

            x = A*rand(T, n)
            y = F\x; @test norm(x - A*y) < approx_rtol*norm(x)

            X = A*rand(T, n, n)
            C = F\X; @test norm(X - A*C) < approx_rtol*norm(X)
        end
      end
    end
  end

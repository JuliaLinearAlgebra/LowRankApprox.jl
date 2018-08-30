#= test/id.jl
=#

  @testset "id" begin
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

            V  = idfact(:n, A, opts)
            F  = ID(:n, A, V)
            Vc = idfact(:c, A, opts)
            Fc = ID(:c, A, Vc)
            @test norm(A  - Matrix(F )) < approx_rtol*nrm
            @test norm(A' - Matrix(Fc)) < approx_rtol*nrm

            for (N, G, ρ, q) in ((Matrix(V), V, V[:k], n), (A, F, m, n))
              xp = rand(T, ρ)
              xq = rand(T, q)
              y = N  *xq; @test norm(y - G  *xq) < approx_rtol*norm(y)
              y = N' *xp; @test norm(y - G' *xp) < approx_rtol*norm(y)
              y = transpose(N)*xp; @test norm(y - transpose(G)*xp) < approx_rtol*norm(y)

              Bp = rand(T, ρ, ρ)
              Bq = rand(T, q, q)
              C = N   *Bq  ; @test norm(C - G   *Bq  ) < approx_rtol*norm(C)
              C = N   *Bq' ; @test norm(C - G   *Bq' ) < approx_rtol*norm(C)
              C = N   *transpose(Bq); @test norm(C - G   *transpose(Bq)) < approx_rtol*norm(C)
              C = N'  *Bp  ; @test norm(C - G'  *Bp  ) < approx_rtol*norm(C)
              C = N'  *Bp' ; @test norm(C - G'  *Bp' ) < approx_rtol*norm(C)
              C = transpose(N) *Bp  ; @test norm(C - transpose(G) *Bp  ) < approx_rtol*norm(C)
              C = transpose(N) *transpose(Bp); @test norm(C - transpose(G) *transpose(Bp)) < approx_rtol*norm(C)
              C = Bp  *N   ; @test norm(C - Bp  *G   ) < approx_rtol*norm(C)
              C = Bq  *N'  ; @test norm(C - Bq  *G'  ) < approx_rtol*norm(C)
              C = Bq  *transpose(N) ; @test norm(C - Bq  *transpose(G) ) < approx_rtol*norm(C)
              C = Bp' *N   ; @test norm(C - Bp' *G   ) < approx_rtol*norm(C)
              C = Bq' *N'  ; @test norm(C - Bq' *G'  ) < approx_rtol*norm(C)
              C = transpose(Bp)*N   ; @test norm(C - transpose(Bp)*G   ) < approx_rtol*norm(C)
              C = transpose(Bq)*transpose(N) ; @test norm(C - transpose(Bq)*transpose(G) ) < approx_rtol*norm(C)
            end
        end
      end
    end
  end

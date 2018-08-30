#= test/cur.jl
=#

  @testset "cur" begin
    m = 128
    n =  64
    M = matrixlib(:fourier, rand(m), rand(n))
    Mh = M[1:n,1:n]; Mh += Mh';
    Ms = M[1:n,1:n]; Ms += transpose(Ms);
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
            approx_rtol = 1000*rtol
            opts.rtol = rtol

            for (N, ρ, q) in ((M, m, n), (Mh, n, n), (Ms, n, n))
              A = convert(Array{T}, T <: Real ? real(N) : N)

              U  = curfact(A, opts)
              F  = CUR(A, U)
              @test norm(A - Matrix(F)) < approx_rtol*norm(A)

              xp = rand(T, ρ)
              xq = rand(T, q)
              y = A  *xq; @test norm(y - F  *xq) < approx_rtol*norm(y)
              y = A' *xp; @test norm(y - F' *xp) < approx_rtol*norm(y)
              y = transpose(A)*xp; @test norm(y - transpose(F)*xp) < approx_rtol*norm(y)

              Bp = rand(T, ρ, ρ)
              Bq = rand(T, q, q)
              C = A   *Bq  ; @test norm(C - F   *Bq  ) < approx_rtol*norm(C)
              C = A   *Bq' ; @test norm(C - F   *Bq' ) < approx_rtol*norm(C)
              C = A   *transpose(Bq); @test norm(C - F   *transpose(Bq)) < approx_rtol*norm(C)
              C = A'  *Bp  ; @test norm(C - F'  *Bp  ) < approx_rtol*norm(C)
              C = A'  *Bp' ; @test norm(C - F'  *Bp' ) < approx_rtol*norm(C)
              C = transpose(A) *Bp  ; @test norm(C - transpose(F) *Bp  ) < approx_rtol*norm(C)
              C = transpose(A) *transpose(Bp); @test norm(C - transpose(F) *transpose(Bp)) < approx_rtol*norm(C)
              C = Bp  *A   ; @test norm(C - Bp  *F   ) < approx_rtol*norm(C)
              C = Bq  *A'  ; @test norm(C - Bq  *F'  ) < approx_rtol*norm(C)
              C = Bq  *transpose(A) ; @test norm(C - Bq  *transpose(F) ) < approx_rtol*norm(C)
              C = Bp' *A   ; @test norm(C - Bp' *F   ) < approx_rtol*norm(C)
              C = Bq' *A'  ; @test norm(C - Bq' *F'  ) < approx_rtol*norm(C)
              C = transpose(Bp)*A   ; @test norm(C - transpose(Bp)*F   ) < approx_rtol*norm(C)
              C = transpose(Bq)*transpose(A) ; @test norm(C - transpose(Bq)*transpose(F) ) < approx_rtol*norm(C)
          end
        end
      end
    end
  end

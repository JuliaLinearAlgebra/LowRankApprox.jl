#= test/pheig.jl
=#

@testset "pheig" begin
  n = 128
  M = matrixlib(:fourier, rand(n), rand(n))
  M += M'
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
          F = pheigfact(A, opts)
          @test norm(A - Matrix(F)) < approx_rtol*norm(A)

          s = pheigvals(A, opts, rank=F[:k], rtol=0.)
          @test norm(s - F[:values]) < approx_rtol*norm(s)

          x = rand(T, n)
          y = A  *x; @test norm(y - F  *x) < approx_rtol*norm(y)
          y = A' *x; @test norm(y - F' *x) < approx_rtol*norm(y)
          y = transpose(A)*x; @test norm(y - transpose(F)*x) < approx_rtol*norm(y)

          X = rand(T, n, n)
          C = A  *X  ; @test norm(C - F  *X  ) < approx_rtol*norm(C)
          C = A  *X' ; @test norm(C - F  *X' ) < approx_rtol*norm(C)
          C = A  *transpose(X); @test norm(C - F  *transpose(X)) < approx_rtol*norm(C)
          C = A' *X  ; @test norm(C - F' *X  ) < approx_rtol*norm(C)
          C = A' *X' ; @test norm(C - F' *X' ) < approx_rtol*norm(C)
          C = transpose(A)*X  ; @test norm(C - transpose(F)*X  ) < approx_rtol*norm(C)
          C = transpose(A)*transpose(X); @test norm(C - transpose(F)*transpose(X)) < approx_rtol*norm(C)
          C = X  *A  ; @test norm(C - X  *F  ) < approx_rtol*norm(C)
          C = X  *A' ; @test norm(C - X  *F' ) < approx_rtol*norm(C)
          C = X  *transpose(A); @test norm(C - X  *transpose(F)) < approx_rtol*norm(C)
          C = X' *A  ; @test norm(C - X' *F  ) < approx_rtol*norm(C)
          C = X' *A' ; @test norm(C - X' *F' ) < approx_rtol*norm(C)
          C = transpose(X)*A  ; @test norm(C - transpose(X)*F  ) < approx_rtol*norm(C)
          C = transpose(X)*transpose(A); @test norm(C - transpose(X)*transpose(F)) < approx_rtol*norm(C)

          x = A*rand(T, n)
          y = F\x; @test norm(x - A*y) < approx_rtol*norm(x)

          X = A*rand(T, n, n)
          C = F\X; @test norm(X - A*C) < approx_rtol*norm(X)
      end
    end
  end
end

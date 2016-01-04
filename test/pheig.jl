#= test/pheig.jl
=#

println("pheig.jl")
tic()

n = 128
M = matrixlib(:fourier, rand(n), rand(n))
M += M'
opts = LRAOptions(rrqr_delta=0., sketch_randn_niter=1)

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
    F = pheigfact(A, opts)
    @test norm(A - full(F)) < approx_rtol*norm(A)

    s = pheigvals(A, opts, rank=F[:k], rtol=0.)
    @test norm(s - F[:values]) < approx_rtol*norm(s)

    x = rand(T, n)
    y = A  *x; @test norm(y - F  *x) < approx_rtol*norm(y)
    y = A' *x; @test norm(y - F' *x) < approx_rtol*norm(y)
    y = A.'*x; @test norm(y - F.'*x) < approx_rtol*norm(y)

    X = rand(T, n, n)
    C = A  *X  ; @test norm(C - F  *X  ) < approx_rtol*norm(C)
    C = A  *X' ; @test norm(C - F  *X' ) < approx_rtol*norm(C)
    C = A  *X.'; @test norm(C - F  *X.') < approx_rtol*norm(C)
    C = A' *X  ; @test norm(C - F' *X  ) < approx_rtol*norm(C)
    C = A' *X' ; @test norm(C - F' *X' ) < approx_rtol*norm(C)
    C = A.'*X  ; @test norm(C - F.'*X  ) < approx_rtol*norm(C)
    C = A.'*X.'; @test norm(C - F.'*X.') < approx_rtol*norm(C)
    C = X  *A  ; @test norm(C - X  *F  ) < approx_rtol*norm(C)
    C = X  *A' ; @test norm(C - X  *F' ) < approx_rtol*norm(C)
    C = X  *A.'; @test norm(C - X  *F.') < approx_rtol*norm(C)
    C = X' *A  ; @test norm(C - X' *F  ) < approx_rtol*norm(C)
    C = X' *A' ; @test norm(C - X' *F' ) < approx_rtol*norm(C)
    C = X.'*A  ; @test norm(C - X.'*F  ) < approx_rtol*norm(C)
    C = X.'*A.'; @test norm(C - X.'*F.') < approx_rtol*norm(C)

    x = A*rand(T, n)
    y = F\x; @test norm(x - A*y) < approx_rtol*norm(x)

    X = A*rand(T, n, n)
    C = F\X; @test norm(X - A*C) < approx_rtol*norm(X)
  end
end

toc()
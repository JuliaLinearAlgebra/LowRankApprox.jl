#= test/peig.jl
=#

println("peig.jl")
tic()

n = 128
M = matrixlib(:fourier, rand(n), rand(n))
rtol = 1e-6
approx_rtol = 100*rtol
opts = LRAOptions(rtol=rtol, sketch_randn_niter=1)

for (t, s) in ((:none,                 :none ),
               (:RandomGaussian,       :randn),
               (:RandomSubset,         :sub  ),
               (:SRFT,                 :srft ),
               (:SparseRandomGaussian, :sprn ))
  opts.sketch = s
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    A = convert(Array{T}, T <: Real ? real(M) : M)

    # PartialEigen

    opts.peig_vecs = :left
    F = peigfact(A, opts)
    B = F[:vectors]'*A
    C = broadcast(*, F[:values], F[:vectors]')
    @test norm(B - C) < approx_rtol*norm(B)

    opts.peig_vecs = :right
    F = peigfact(A, opts)
    B = A*F[:vectors]
    C = broadcast(*, F[:vectors], F[:values].')
    @test norm(B - C) < approx_rtol*norm(B)

    opts_ = copy(opts, rank=F[:k], rtol=0.)
    s = peigvals(A, opts_)
    @test norm(s - F[:values]) < approx_rtol*norm(s)

    # HermPartialEigen

    A += A'
    F = peigfact(A, opts)
    @test norm(A - full(F)) < approx_rtol*norm(A)

    opts_ = copy(opts, rank=F[:k], rtol=0.)
    s = peigvals(A, opts_)
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
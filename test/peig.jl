#= test/peig.jl
=#

import LowRankApprox.hilbert

println("peig.jl")
tic()

n = 128
A = hilbert(n, n)*(1 + im)
rtol = 1e-6
approx_rtol = 100*rtol
opts = LRAOptions(rtol=rtol, sketch_randn_niter=1)

for (t, s) in ((:none,                 :none),
               (:RandomGaussian,       :randn),
               (:RandomSubset,         :subs),
               (:SRFT,                 :srft),
               (:SparseRandomGaussian, :sprn))
  opts.sketch = s
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    B = T <: Real ? real(A) : A
    B = convert(Array{T}, B)

    # PartialEigen

    opts.peig_vecs = :left
    F = peigfact(B, opts)
    C = F[:vectors]'*B
    D = broadcast(*, F[:values], F[:vectors]')
    @test_approx_eq_eps C D approx_rtol*snorm(C)

    opts.peig_vecs = :right
    F = peigfact(B, opts)
    C = B*F[:vectors]
    D = broadcast(*, F[:vectors], F[:values].')
    @test_approx_eq_eps C D approx_rtol*snorm(C)

    opts_ = copy(opts, rank=F[:k], rtol=0.)
    s = peigvals(B, opts_)
    @test_approx_eq_eps s F[:values] approx_rtol*norm(s)

    # HermPartialEigen

    B += flipdim(B, 1)
    B += B'
    F = peigfact(B, opts)
    @test_approx_eq_eps B full(F) approx_rtol*snorm(B)

    opts_ = copy(opts, rank=F[:k], rtol=0.)
    s = peigvals(B, opts_)
    @test_approx_eq_eps s F[:values] approx_rtol*norm(s)

    x = rand(T, n)
    y = B  *x; @test_approx_eq_eps y F  *x approx_rtol*norm(y)
    y = B' *x; @test_approx_eq_eps y F' *x approx_rtol*norm(y)
    y = B.'*x; @test_approx_eq_eps y F.'*x approx_rtol*norm(y)

    X = rand(T, n, n)
    C = B  *X  ; @test_approx_eq_eps C F  *X   approx_rtol*snorm(C)
    C = B  *X' ; @test_approx_eq_eps C F  *X'  approx_rtol*snorm(C)
    C = B  *X.'; @test_approx_eq_eps C F  *X.' approx_rtol*snorm(C)
    C = B' *X  ; @test_approx_eq_eps C F' *X   approx_rtol*snorm(C)
    C = B' *X' ; @test_approx_eq_eps C F' *X'  approx_rtol*snorm(C)
    C = B.'*X  ; @test_approx_eq_eps C F.'*X   approx_rtol*snorm(C)
    C = B.'*X.'; @test_approx_eq_eps C F.'*X.' approx_rtol*snorm(C)
    C = X  *B  ; @test_approx_eq_eps C X  *F   approx_rtol*snorm(C)
    C = X  *B' ; @test_approx_eq_eps C X  *F'  approx_rtol*snorm(C)
    C = X  *B.'; @test_approx_eq_eps C X  *F.' approx_rtol*snorm(C)
    C = X' *B  ; @test_approx_eq_eps C X' *F   approx_rtol*snorm(C)
    C = X' *B' ; @test_approx_eq_eps C X' *F'  approx_rtol*snorm(C)
    C = X.'*B  ; @test_approx_eq_eps C X.'*F   approx_rtol*snorm(C)
    C = X.'*B.'; @test_approx_eq_eps C X.'*F.' approx_rtol*snorm(C)

    x = B*rand(T, n)
    y = F\x; @test_approx_eq_eps x B*y approx_rtol*norm(y)

    X = B*rand(T, n, n)
    C = F\X; @test_approx_eq_eps X B*C approx_rtol*snorm(X)
  end
end

toc()
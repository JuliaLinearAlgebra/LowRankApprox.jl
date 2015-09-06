#= test/id.jl
=#

import LowRankApprox.hilbert

println("id.jl")
tic()

m = 128
n = 64
A = hilbert(m, n)*(1 + im)
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

    V = idfact(B, opts)
    F = ID(B, V)
    @test_approx_eq_eps B full(F) approx_rtol*snorm(B)

    xm = rand(T, m)
    xn = rand(T, n)
    y = B  *xn; @test_approx_eq_eps y F  *xn approx_rtol*norm(y)
    y = B' *xm; @test_approx_eq_eps y F' *xm approx_rtol*norm(y)
    y = B.'*xm; @test_approx_eq_eps y F.'*xm approx_rtol*norm(y)

    Bm = rand(T, m, m)
    Bn = rand(T, n, n)
    C = B   *Bn  ; @test_approx_eq_eps C F   *Bn   approx_rtol*snorm(C)
    C = B   *Bn' ; @test_approx_eq_eps C F   *Bn'  approx_rtol*snorm(C)
    C = B   *Bn.'; @test_approx_eq_eps C F   *Bn.' approx_rtol*snorm(C)
    C = B'  *Bm  ; @test_approx_eq_eps C F'  *Bm   approx_rtol*snorm(C)
    C = B'  *Bm' ; @test_approx_eq_eps C F'  *Bm'  approx_rtol*snorm(C)
    C = B.' *Bm  ; @test_approx_eq_eps C F.' *Bm   approx_rtol*snorm(C)
    C = B.' *Bm.'; @test_approx_eq_eps C F.' *Bm.' approx_rtol*snorm(C)
    C = Bm  *B   ; @test_approx_eq_eps C Bm  *F    approx_rtol*snorm(C)
    C = Bn  *B'  ; @test_approx_eq_eps C Bn  *F'   approx_rtol*snorm(C)
    C = Bn  *B.' ; @test_approx_eq_eps C Bn  *F.'  approx_rtol*snorm(C)
    C = Bm' *B   ; @test_approx_eq_eps C Bm' *F    approx_rtol*snorm(C)
    C = Bn' *B'  ; @test_approx_eq_eps C Bn' *F'   approx_rtol*snorm(C)
    C = Bm.'*B   ; @test_approx_eq_eps C Bm.'*F    approx_rtol*snorm(C)
    C = Bn.'*B.' ; @test_approx_eq_eps C Bn.'*F.'  approx_rtol*snorm(C)
  end
end

toc()
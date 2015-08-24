#= test/pqr.jl
=#

import LowRankApprox.hilbert

println("pqr.jl")
tic()

m = 20
n = 10
A = hilbert(m, n)
rtol = 1e-6

for T in (Float32, Float64, Complex64, Complex128)
  println("  $T")

  if T <: Complex
    A *= 1 + im
  end

  F = pqrfact(A, rtol)
  @test_approx_eq_eps A full(F) norm(A)*rtol

  xm = rand(T, m)
  xn = rand(T, n)
  y = A  *xn; @test_approx_eq_eps y F  *xn rtol*norm(y)
  y = A' *xm; @test_approx_eq_eps y F' *xm rtol*norm(y)
  y = A.'*xm; @test_approx_eq_eps y F.'*xm rtol*norm(y)

  Bm = rand(T, m, m)
  Bn = rand(T, n, n)
  C = A   *Bn  ; @test_approx_eq_eps C F   *Bn   rtol*norm(C)
  C = A   *Bn' ; @test_approx_eq_eps C F   *Bn'  rtol*norm(C)
  C = A   *Bn.'; @test_approx_eq_eps C F   *Bn.' rtol*norm(C)
  C = A'  *Bm  ; @test_approx_eq_eps C F'  *Bm   rtol*norm(C)
  C = A'  *Bm' ; @test_approx_eq_eps C F'  *Bm'  rtol*norm(C)
  C = A.' *Bm  ; @test_approx_eq_eps C F.' *Bm   rtol*norm(C)
  C = A.' *Bm.'; @test_approx_eq_eps C F.' *Bm.' rtol*norm(C)
  C = Bm  *A   ; @test_approx_eq_eps C Bm  *F    rtol*norm(C)
  C = Bn  *A'  ; @test_approx_eq_eps C Bn  *F'   rtol*norm(C)
  C = Bn  *A.' ; @test_approx_eq_eps C Bn  *F.'  rtol*norm(C)
  C = Bm' *A   ; @test_approx_eq_eps C Bm' *F    rtol*norm(C)
  C = Bn' *A'  ; @test_approx_eq_eps C Bn' *F'   rtol*norm(C)
  C = Bm.'*A   ; @test_approx_eq_eps C Bm.'*F    rtol*norm(C)
  C = Bn.'*A.' ; @test_approx_eq_eps C Bn.'*F.'  rtol*norm(C)
end

toc()
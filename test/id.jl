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

    for (p, q, M, S) in ((V[:k], n, full(V), V), (m, n, B, F))
      xp = rand(T, p)
      xq = rand(T, q)
      y = M  *xq; @test_approx_eq_eps y S  *xq approx_rtol*norm(y)
      y = M' *xp; @test_approx_eq_eps y S' *xp approx_rtol*norm(y)
      y = M.'*xp; @test_approx_eq_eps y S.'*xp approx_rtol*norm(y)

      Bp = rand(T, p, p)
      Bq = rand(T, q, q)
      C = M   *Bq  ; @test_approx_eq_eps C S   *Bq   approx_rtol*snorm(C)
      C = M   *Bq' ; @test_approx_eq_eps C S   *Bq'  approx_rtol*snorm(C)
      C = M   *Bq.'; @test_approx_eq_eps C S   *Bq.' approx_rtol*snorm(C)
      C = M'  *Bp  ; @test_approx_eq_eps C S'  *Bp   approx_rtol*snorm(C)
      C = M'  *Bp' ; @test_approx_eq_eps C S'  *Bp'  approx_rtol*snorm(C)
      C = M.' *Bp  ; @test_approx_eq_eps C S.' *Bp   approx_rtol*snorm(C)
      C = M.' *Bp.'; @test_approx_eq_eps C S.' *Bp.' approx_rtol*snorm(C)
      C = Bp  *M   ; @test_approx_eq_eps C Bp  *S    approx_rtol*snorm(C)
      C = Bq  *M'  ; @test_approx_eq_eps C Bq  *S'   approx_rtol*snorm(C)
      C = Bq  *M.' ; @test_approx_eq_eps C Bq  *S.'  approx_rtol*snorm(C)
      C = Bp' *M   ; @test_approx_eq_eps C Bp' *S    approx_rtol*snorm(C)
      C = Bq' *M'  ; @test_approx_eq_eps C Bq' *S'   approx_rtol*snorm(C)
      C = Bp.'*M   ; @test_approx_eq_eps C Bp.'*S    approx_rtol*snorm(C)
      C = Bq.'*M.' ; @test_approx_eq_eps C Bq.'*S.'  approx_rtol*snorm(C)
    end
  end
end

toc()
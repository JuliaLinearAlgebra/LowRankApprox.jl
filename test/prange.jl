#= test/prange.jl
=#

import LowRankApprox.hilbert

println("prange.jl")
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

    nrm = snorm(B)
    Q = prange(:n,  B, opts)
    @test_approx_eq_eps B Q*(Q'*B)      approx_rtol*nrm
    Q = prange(:c,  B, opts)
    @test_approx_eq_eps B      (B*Q)*Q' approx_rtol*nrm
    Q = prange(:nc, B, opts)
    @test_approx_eq_eps B Q*(Q'*B*Q)*Q' approx_rtol*nrm
  end
end

toc()
#= test/rrange.jl
=#

import LowRankApprox.hilbert

println("rrange.jl")
tic()

m = 128
n = 64
A = hilbert(m, n)*(1 + im)
rtol = 1e-6
approx_rtol = 100*rtol
opts = LRAOptions(rtol=rtol, sketch_randn_niter=1)

for (t, s) in ((:RandomGaussian,       :randn),
               (:RandomSubset,         :subs),
               (:SRFT,                 :srft),
               (:SparseRandomGaussian, :sprn))
  opts.sketch = s
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    B = T <: Real ? real(A) : A
    B = convert(Array{T}, B)

    nrm = norm(B)
    Q = rrange(:n, B, opts)
    @test_approx_eq_eps B Q*(Q'*B) approx_rtol*nrm
    Q = rrange(:c, B, opts)
    @test_approx_eq_eps B (B*Q)*Q' approx_rtol*nrm
  end
end

toc()
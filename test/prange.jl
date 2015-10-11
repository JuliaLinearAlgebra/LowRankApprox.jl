#= test/prange.jl
=#

println("prange.jl")
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
    nrm = norm(A)

    Q = prange(:n, A, opts)
    @test norm(A - Q*(Q'*A)     ) < approx_rtol*nrm
    Q = prange(:c, A, opts)
    @test norm(A -      (A*Q)*Q') < approx_rtol*nrm
    Q = prange(:b, A, opts)
    @test norm(A - Q*(Q'*A*Q)*Q') < approx_rtol*nrm
  end
end

toc()
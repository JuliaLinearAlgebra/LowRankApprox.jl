#= test/trapezoidal.jl
=#

println("trapezoidal.jl")
tic()

m = 20
n = 10
data_lo = rand(Complex128, m, n)
data_up = data_lo'

for (t, data, trilu) in ((:LowerTrapezoidal, :data_lo, :tril),
                         (:UpperTrapezoidal, :data_up, :triu))
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    @eval A = $t{$T}($T <: Real ? real($data) : $data)
    m, n = size(A)
    F = full(A)
    @test_approx_eq (@eval $trilu(A.data)) F

    xm = rand(T, m)
    xn = rand(T, n)
    @test_approx_eq A  *xn F  *xn
    @test_approx_eq A' *xm F' *xm
    @test_approx_eq A.'*xm F.'*xm

    Bm = rand(T, m, m)
    Bn = rand(T, n, n)
    @test_approx_eq A   *Bn   F   *Bn
    @test_approx_eq A   *Bn'  F   *Bn'
    @test_approx_eq A   *Bn.' F   *Bn.'
    @test_approx_eq A'  *Bm   F'  *Bm
    @test_approx_eq A'  *Bm'  F'  *Bm'
    @test_approx_eq A.' *Bm   F.' *Bm
    @test_approx_eq A.' *Bm.' F.' *Bm.'
    @test_approx_eq Bm  *A    Bm  *F
    @test_approx_eq Bn  *A'   Bn  *F'
    @test_approx_eq Bn  *A.'  Bn  *F.'
    @test_approx_eq Bm' *A    Bm' *F
    @test_approx_eq Bn' *A'   Bn' *F'
    @test_approx_eq Bm.'*A    Bm.'*F
    @test_approx_eq Bn.'*A.'  Bn.'*F.'
  end
end

toc()
#= test/trapezoidal.jl
=#

println("trapezoidal.jl")

m = 20
n = 10
data_up = rand(ComplexF64, m, n)
data_lo = data_up'

@time for (t, data, trilu) in ((:LowerTrapezoidal, :data_up, :tril),
                               (:UpperTrapezoidal, :data_lo, :triu))
  for T in (Float32, Float64, ComplexF32, ComplexF64)
    let n, m
        println("  $t/$T")

        @eval A = $t{$T}($T <: Real ? real($data) : $data)
        m, n = size(A)
        F = full(A)
        @test (@eval $trilu(A.data)) ≈ F

        xm = rand(T, m)
        xn = rand(T, n)
        @test A  *xn ≈ F  *xn
        @test A' *xm ≈ F' *xm
        @test A.'*xm ≈ F.'*xm

        Bm = rand(T, m, m)
        Bn = rand(T, n, n)
        @test A   *Bn   ≈ F   *Bn
        @test A   *Bn'  ≈ F   *Bn'
        @test A   *Bn.' ≈ F   *Bn.'
        @test A'  *Bm   ≈ F'  *Bm
        @test A'  *Bm'  ≈ F'  *Bm'
        @test A.' *Bm   ≈ F.' *Bm
        @test A.' *Bm.' ≈ F.' *Bm.'
        @test Bm  *A    ≈ Bm  *F
        @test Bn  *A'   ≈ Bn  *F'
        @test Bn  *A.'  ≈ Bn  *F.'
        @test Bm' *A    ≈ Bm' *F
        @test Bn' *A'   ≈ Bn' *F'
        @test Bm.'*A    ≈ Bm.'*F
        @test Bn.'*A.'  ≈ Bn.'*F.'
    end
  end
end

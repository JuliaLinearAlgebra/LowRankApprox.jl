#= test/trapezoidal.jl
=#

@testset "trapezoidal" begin
  m = 20
  n = 10
  data_up = rand(ComplexF64, m, n)
  data_lo = data_up'

  @time for (t, data, trilu) in ((LowerTrapezoidal, data_up, tril),
                                 (UpperTrapezoidal, data_lo, triu))
    for T in (Float32, Float64, ComplexF32, ComplexF64)
      let n, m
          println("  $t/$T")

          A = t{T}(T <: Real ? real(data) : data)
          m, n = size(A)
          F = Matrix(A)
          @test trilu(A.data) ≈ F

          xm = rand(T, m)
          xn = rand(T, n)
          @test A  *xn ≈ F  *xn
          @test A' *xm ≈ F' *xm
          @test transpose(A)*xm ≈ transpose(F)*xm

          Bm = rand(T, m, m)
          Bn = rand(T, n, n)
          @test A   *Bn   ≈ F   *Bn
          @test A   *Bn'  ≈ F   *Bn'
          @test A   *transpose(Bn) ≈ F   *transpose(Bn)
          @test A'  *Bm   ≈ F'  *Bm
          @test A'  *Bm'  ≈ F'  *Bm'
          @test transpose(A) *Bm   ≈ transpose(F) *Bm
          @test transpose(A) *transpose(Bm) ≈ transpose(F) *transpose(Bm)
          @test Bm  *A    ≈ Bm  *F
          @test Bn  *A'   ≈ Bn  *F'
          @test Bn  *transpose(A)  ≈ Bn  *transpose(F)
          @test Bm' *A    ≈ Bm' *F
          @test Bn' *A'   ≈ Bn' *F'
          @test transpose(Bm)*A    ≈ transpose(Bm)*F
          @test transpose(Bn)*transpose(A)  ≈ transpose(Bn)*transpose(F)
      end
    end
  end
end

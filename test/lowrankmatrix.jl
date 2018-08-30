using LowRankApprox, FillArrays, Test

@testset "LowRankMatrix" begin
    @testset "Constructors" begin
        @test Matrix(LowRankMatrix(Zeros(10,5))) == zeros(10,5)

        @test LowRankMatrix{Float64}(Zeros(10,5)) == LowRankMatrix(Zeros(10,5)) ==
                    LowRankMatrix{Float64}(Zeros(10,5),1) == LowRankMatrix{Float64}(Zeros{Int}(10,5),1) ==
                    LowRankMatrix{Float64}(zeros(10,5)) == LowRankMatrix(zeros(10,5))  ==
                    LowRankMatrix{Float64}(zeros(Int,10,5))


        @test isempty(LowRankMatrix{Float64}(zeros(10,5)).U)
        @test isempty(LowRankMatrix{Float64}(zeros(10,5)).V)
        @test isempty(LowRankMatrix(Zeros(10,5)).U)
        @test isempty(LowRankMatrix(Zeros(10,5)).V)

        @test rank(LowRankMatrix(Zeros(10,5))) == 0


        @test Matrix(LowRankMatrix(Ones(10,5))) == fill(1.0,10,5)
        @test LowRankMatrix{Float64}(Ones(10,5)) == LowRankMatrix(Ones(10,5)) ==
                    LowRankMatrix{Float64}(Ones{Int}(10,5))
        @test LowRankMatrix{Float64}(fill(1.0,10,5)) == LowRankMatrix(fill(1.0,10,5))  ==
                    LowRankMatrix{Float64}(fill(1,10,5))
        @test rank(LowRankMatrix(Ones(10,5))) == 1

        @test LowRankMatrix(Ones(10,5)) ≈ LowRankMatrix(fill(1.0,10,5))


        @test Matrix(LowRankMatrix(Ones(10,5))) == fill(1.0,10,5)
        @test LowRankMatrix{Float64}(Ones(10,5)) == LowRankMatrix(Ones(10,5))
        @test rank(LowRankMatrix(fill(1.0,10,5))) == 1

        x = 2
        @test Matrix(LowRankMatrix(Fill(x,10,5))) ≈ fill(x,10,5)
        @test LowRankMatrix{Float64}(Fill(x,10,5)) == LowRankMatrix(Fill(x,10,5)) ==
                    LowRankMatrix{Float64}(Fill{Float64}(x,10,5))
        @test LowRankMatrix{Float64}(fill(x,10,5)) == LowRankMatrix(fill(x,10,5))  ==
                    LowRankMatrix{Float64}(fill(x,10,5))
        @test rank(LowRankMatrix(Fill(x,10,5))) == 1


    end


    @testset "LowRankMatrix algebra" begin
        A = LowRankApprox._LowRankMatrix(randn(20,4), randn(12,4))
        @test Matrix(2*A) ≈ Matrix(A*2) ≈ 2*Matrix(A)

        B = LowRankApprox._LowRankMatrix(randn(20,2), randn(12,2))
        @test Matrix(A+B) ≈ Matrix(A) + Matrix(B) ≈ Matrix(Matrix(A) + B) ≈
                    Matrix(A + Matrix(B))
        @test rank(A+B) ≤ rank(A) + rank(B)

        @test Matrix(A-B) ≈ Matrix(A) - Matrix(B) ≈ Matrix(Matrix(A) - B) ≈
                    Matrix(A - Matrix(B))
        @test rank(A-B) ≤ rank(A) + rank(B)

        B = LowRankApprox._LowRankMatrix(randn(12,2), randn(14,2))

        @test A*B isa LowRankMatrix
        @test rank(A*B) == size((A*B).U,2) == 4

        @test Matrix(A)*Matrix(B) ≈ Matrix(A*Matrix(B)) ≈ Matrix(Matrix(A)*B) ≈ Matrix(A*B)

        B = LowRankApprox._LowRankMatrix(randn(10,2), randn(14,2))
        @test_throws DimensionMismatch A*B
        @test_throws DimensionMismatch B*A

        B = randn(12,14)
        @test A*B isa LowRankMatrix
        @test rank(A*B) == size((A*B).U,2) == 4
        @test Matrix(A)*B ≈ Matrix(A*B)

        B = randn(20,20)
        @test B*A isa LowRankMatrix
        @test rank(B*A) == size((B*A).U,2) == 4
        @test B*Matrix(A) ≈ Matrix(B*A)

        v = randn(12)
        @test all(LowRankApprox.mul!(randn(size(A,1)), A, v) .=== A*v )
        @test A*v ≈ Matrix(A)*v
    end
end

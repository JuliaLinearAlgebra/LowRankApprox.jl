#= src/util.jl
=#

function getcols(trans::Symbol, A::AbstractMatrix, cols)
  if     trans == :n  return A[:,cols]
  elseif trans == :c  return A[cols,:]'
  end
end
function getcols(trans::Symbol, A::AbstractLinearOperator, cols)
  if     trans == :n  return A[:,cols]
  elseif trans == :c  return A'[:, cols]
  end
end

symrelerr(x, y) = 2*abs((x - y)/(x + y))
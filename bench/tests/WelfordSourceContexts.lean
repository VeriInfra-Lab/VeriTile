import VeriTile.Triton

namespace VeriTile.Tests.WelfordSourceContexts
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000

def selected : DSL.WelfordSource.Tree 2 := .node (.leaf 0) (.leaf 1)

def tupleSource : ComputeKernel := triton {
  mean = tl.zeros([1, 2], tl.float32)
  m2 = tl.zeros([1, 2], tl.float32)
  weight = tl.zeros([1, 2], tl.float32)
  resultMean, resultM2, resultWeight = triton_helpers.welford(mean, m2, weight, 1, tree=$(selected))
}

example : tupleSource.surfaceBody.drop 3 =
    [.ifThen (.constBool Bool.true)
      [.assign .real [1] "resultMean" (.compute (.alg .fp32 (.full [1]
          (DSL.WelfordSource.expressions (.ref .real [1,2] "mean")
            (.ref .real [1,2] "m2") (.ref .real [1,2] "weight") selected).mean))),
       .assign .real [1] "resultM2" (.compute (.alg .fp32 (.full [1]
          (DSL.WelfordSource.expressions (.ref .real [1,2] "mean")
            (.ref .real [1,2] "m2") (.ref .real [1,2] "weight") selected).m2))),
       .assign .real [1] "resultWeight" (.compute (.alg .fp32 (.full [1]
          (DSL.WelfordSource.expressions (.ref .real [1,2] "mean")
            (.ref .real [1,2] "m2") (.ref .real [1,2] "weight") selected).weight)))]] := by rfl

def sumSource : ComputeKernel := triton {
  input = tl.zeros([1, 2], tl.float32)
  total = tl.sum(input, axis=1, tree=$(selected), keep_dims=true)
}

example : sumSource.surfaceBody.drop 1 =
    [.assign .real [1,1] "total" (.compute (.alg .fp32 (.full [1,1]
      (DSL.WelfordSource.sumExpression (.ref .real [1,2] "input") selected))))] := by rfl

/-- error: welford requires [1, N] tiles and axis 1 -/
#guard_msgs in
example : ComputeKernel := triton {
  mean = tl.zeros([1, 2], tl.float32)
  m2 = tl.zeros([1, 2], tl.float32)
  weight = tl.zeros([1, 2], tl.float32)
  a, b, c = triton_helpers.welford(mean, m2, weight, 0, tree=$(selected))
}

/-- error: welford requires explicit FP32 operands -/
#guard_msgs in
example : ComputeKernel := triton {
  mean = tl.zeros([1, 2], tl.float64)
  m2 = tl.zeros([1, 2], tl.float64)
  weight = tl.zeros([1, 2], tl.float64)
  a, b, c = triton_helpers.welford(mean, m2, weight, 1, tree=$(selected))
}

/-- error: welford requires three distinct targets -/
#guard_msgs in
example : ComputeKernel := triton {
  mean = tl.zeros([1, 2], tl.float32)
  m2 = tl.zeros([1, 2], tl.float32)
  weight = tl.zeros([1, 2], tl.float32)
  a, a, c = triton_helpers.welford(mean, m2, weight, 1, tree=$(selected))
}

/-- error: welford: shape mismatch -/
#guard_msgs in
example : ComputeKernel := triton {
  mean = tl.zeros([1, 2], tl.float32)
  m2 = tl.zeros([1, 3], tl.float32)
  weight = tl.zeros([1, 2], tl.float32)
  a, b, c = triton_helpers.welford(mean, m2, weight, 1, tree=$(selected))
}

/-- error: welford_reduce: dtype mismatch -/
#guard_msgs in
example : ComputeKernel := triton {
  x = tl.full([1, 2], $((0 : Nat)), dtype=tl.int64)
  mean = tl.zeros([1, 2], tl.float32)
  m2 = tl.zeros([1, 2], tl.float32)
  weight = tl.zeros([1, 2], tl.float32)
  a, b, c = triton_helpers.welford_reduce(x, mean, m2, weight, $(Bool.true))
}

/-- error: welford targets must be separate from its input expressions -/
#guard_msgs in
example : ComputeKernel := triton {
  mean = tl.zeros([1, 2], tl.float32)
  m2 = tl.zeros([1, 2], tl.float32)
  weight = tl.zeros([1, 2], tl.float32)
  mean, b, c = triton_helpers.welford(mean, m2, weight, 1, tree=$(selected))
}

/-- error: welford_reduce targets must be separate from its input expressions -/
#guard_msgs in
example : ComputeKernel := triton {
  x = tl.zeros([1, 2], tl.float32)
  mean = tl.zeros([1, 2], tl.float32)
  m2 = tl.zeros([1, 2], tl.float32)
  weight = tl.zeros([1, 2], tl.float32)
  mean, b, c = triton_helpers.welford_reduce(x, mean, m2, weight, $(Bool.true))
}

end
end VeriTile.Tests.WelfordSourceContexts

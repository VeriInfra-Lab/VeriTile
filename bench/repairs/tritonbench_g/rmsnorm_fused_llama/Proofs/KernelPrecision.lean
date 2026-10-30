import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Kernels
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Precision

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelPrecision
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

/-- The same complete quotation differs only at the scalar epsilon binding. -/
theorem shared_body (X Y W : RegionName) (N B : Nat) (eps : ℝ) :
    (runtimeKernel X Y W N B).surfaceBody.take 6=(kernel X Y W N B eps).surfaceBody.take 6 ∧
    (runtimeKernel X Y W N B).surfaceBody.drop 7=(kernel X Y W N B eps).surfaceBody.drop 7 ∧
    (runtimeKernel X Y W N B).surfaceBody.length=8 := ⟨rfl,rfl,rfl⟩

/-- Runtime epsilon is read from an FP32 scalar register, never a literal. -/
theorem runtime_epsilon (X Y W : RegionName) (N B : Nat) :
    (runtimeKernel X Y W N B).surfaceBody[6]! =
      .assign .real [] "rstd" (.compute (.alg .fp32
        (.div .real .nil (.const 1) (.sqrt (.add .real .nil
          (.ref .real [] "var") (.ref .real [] "eps")))))) := rfl

/-- Source load casts and the FP32→FP16→FP32 store are exact typed AST facts. -/
theorem runtime_memory (X Y W : RegionName) (N B : Nat) :
    (loopBody (runtimeKernel X Y W N B).surfaceBody[4]!)[1]! =inputCast .fp32 Bool.true N B ∧
    (loopBody (runtimeKernel X Y W N B).surfaceBody[7]!)[2]! =weightLoad .fp32 W B ∧
    (loopBody (runtimeKernel X Y W N B).surfaceBody[7]!)[3]! =inputCast .fp32 Bool.false N B ∧
    (loopBody (runtimeKernel X Y W N B).surfaceBody[7]!)[6]! =store .fp32 B := ⟨rfl,rfl,rfl,rfl⟩

theorem runtime_roles (X Y W : RegionName) (N B : Nat) :
    (runtimeKernel X Y W N B).inputs=[X,W] ∧
    (runtimeKernel X Y W N B).outputs=[Y] := ⟨rfl,rfl⟩

#axiomsClean shared_body
#axiomsClean runtime_epsilon
#axiomsClean runtime_memory
#axiomsClean runtime_roles
end VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelPrecision

import bench.tritonbench_g.rotary_emb_nopad.Kernels
import bench.tritonbench_g.rotary_emb.Proofs.TypedExecution

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Ops := VeriTile.Bench.Repairs.RotaryEmb.Ops
abbrev algebra {A : Type} (rest : Algebra A) (ops : Ops A) :=
  VeriTile.Bench.Repairs.RotaryEmb.Typed.algebra rest ops

/-- Complete uncached source, including shared coefficient loads, both Q
stores and the conditional GQA-leader K branch. -/
def uncachedExecution {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s : State A) :
    {t : State A // FP.Structural.exec (algebra rest ops) (Kernels.uncached p) s=some t} := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  have signedLeader :
      ((s.pids 0 : Int)-(p.KV_GROUP_NUM : Int)*((s.pids 0 : Int)/(p.KV_GROUP_NUM : Int))=0) ↔
        s.pids 0%p.KV_GROUP_NUM=0 := by
    rw [←Int.emod_def,←Int.natCast_emod,Int.natCast_eq_zero]
  by_cases leader : s.pids 0 % p.KV_GROUP_NUM=0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [Kernels.uncached,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
      algebra,VeriTile.Bench.Repairs.RotaryEmb.Typed.algebra,
      numericLt,natLt,numericEq,bop,numeric,store,VeriTile.Triton.Region.cast,
      ComputeDType.eraseDType,ofFloat,toFloat,integralFloorDiv,integralMod,
      headNonneg,signedLeader,leader]
    rfl

/-- Complete cached source, retaining the early head return, both signed
metadata loads, cache stores before Q stores, and the K-head masks. -/
def cachedExecution {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.CachedLayout) (s : State A) :
    {t : State A // FP.Structural.exec (algebra rest ops) (Kernels.cached p) s=some t} := by
  by_cases head : s.pids 0<p.Q_HEAD_NUM
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [Kernels.cached,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
      algebra,VeriTile.Bench.Repairs.RotaryEmb.Typed.algebra,
      numericLt,natLt,numericEq,bop,numeric,store,VeriTile.Triton.Region.cast,
      ComputeDType.eraseDType,ofFloat,toFloat,integralFloorDiv,integralMod,head]
    rfl

#axiomsClean uncachedExecution
#axiomsClean cachedExecution
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed

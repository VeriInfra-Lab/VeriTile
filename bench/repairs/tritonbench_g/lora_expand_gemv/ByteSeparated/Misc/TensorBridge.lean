import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Checked
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Guard
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Real
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Access
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.OutputFrame
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Scheduling
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.KernelRealResults
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.KernelNativeResults

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.TensorBridge
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
noncomputable section

theorem checked_real (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (spans : Backings) (indices : Nat→Int)
    (ok : Admitted t s spans indices) (data : RegionName→Nat→ℝ)
    (order : List Nat) (coverage : order.Perm (List.range (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count)) :
    ∃ final,execute rest plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) order (entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) indices)=some final ∧
      (∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w →
        Real.eval data ((final "Y" (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).read .real)=
          Real.target data (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) indices b n) ∧
      (∀ r off,(r≠("Y" : RegionName) ∨ ∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w → off≠yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n) →
        final r off=entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) indices r off) := by
  exact KernelRealResults.kernel_real rest plans
    (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) indices
    (valid t s spans indices ok) data order coverage

/-- Retained compiler realization boundary: shape-only choices and six local
scalar instruction families. No whole kernel/result equality is assumed.
The production pin names the separate candidate, never the frozen wrapper. -/
structure CompilerBinding (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) where
  sourceAST : String
  sourceIdentity : sourceAST=productionAST
  compilerPlans : ∀ n,ReductionPlan n
  planIdentity : compilerPlans=plans
  compilerSchedule : Native.Schedule
  scheduleIdentity : compilerSchedule=schedule
  primitives : Native.ScalarOps
  localAgreement : Native.ScalarLaws primitives ops
  narrowZero : primitives.narrow 0=0
  widenZero : primitives.widen 0=0

abbrev nativeReference := reference

theorem checked_fp (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) (t : Inputs) (s : Scalars) (spans : Backings)
    (before : Raw.State) (ok : Admitted t s spans (indices t before))
    (order : List Nat) (coverage : order.Perm (List.range (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count)) :
    execute rest plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) order (entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before))=
      some (finalMemory rest plans t s before) ∧
    (∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w →
      Native.eval ops schedule (bits t before)
        ((finalMemory rest plans t s before "Y" (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).read .real)=
          nativeReference ops schedule plans t s before b n) := by
  refine ⟨runtime_schedule rest plans t s before ok order coverage,?_⟩
  intro b n hb hn
  rw [(complete rest plans t s before ok).2.1 b n hb hn]
  rfl

/-- Complete original kernel success, every active output byte, complete X/W/
metadata backing frames, and every other byte. The ordinary Tensor binding
contains actual core FP32 X/Y objects; separation is derived from this
candidate's interval check, not from StorageImpl identity. -/
theorem checked_caller_bytes (rest : Algebra Word) (ops : Native.ScalarOps)
    (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (compiler : CompilerBinding ops schedule plans) (t : Inputs) (s : Scalars)
    (spans : Backings) (a : Binding t spans) (before : Raw.State)
    (ok : Admitted t s spans (indices t before)) (order : List Nat)
    (coverage : order.Perm (List.range (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count)) :
    execute rest compiler.compilerPlans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) order (entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before))=
      some (finalMemory rest compiler.compilerPlans t s before) ∧
    (∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w → indices t before b≠-1 → ∀ byte : Fin 4,
      sourceStores rest compiler.primitives compiler.compilerSchedule compiler.compilerPlans t s before
        t.y.device.index ((tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).address+byte.val)=
          Raw.unpack (nativeReference ops schedule plans t s before b n) byte) ∧
    (∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w →
      indices t before b=-1 → ∀ byte : Fin 4,
      sourceStores rest compiler.primitives compiler.compilerSchedule compiler.compilerPlans t s before
        t.y.device.index ((tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).address+byte.val)=
      before t.y.device.index ((tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).address+byte.val)) ∧
    (∀ owner address,spans.x.covers address ∨ spans.w.covers address ∨ spans.indices.covers address →
      sourceStores rest compiler.primitives compiler.compilerSchedule compiler.compilerPlans t s before
        owner address=before owner address) ∧
    (∀ owner address,(∀ q,q<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count → indices t before (q/(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).N)≠-1 →
      ¬(key t s q).covers owner address) →
      sourceStores rest compiler.primitives compiler.compilerSchedule compiler.compilerPlans t s before
        owner address=before owner address) := by
  rw [compiler.planIdentity,compiler.scheduleIdentity,
    Native.operations_eq compiler.primitives ops compiler.localAgreement]
  exact ⟨runtime_schedule rest plans t s before ok order coverage,
    (whole rest ops schedule plans t s before ok).2.1,
    sentinel_frame rest ops schedule plans t s spans before ok,
    caller_frame rest ops schedule plans t s spans a before ok,
    source_frame rest ops schedule plans t s before⟩

#axiomsClean checked_real
#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
#specNonCircular nativeReference avoiding [execution,finalMemory,sourceStores,execute,Branches.source,FP.Structural.exec]
#specNonCircular CompilerBinding avoiding [execution,finalMemory,sourceStores,execute,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.TensorBridge

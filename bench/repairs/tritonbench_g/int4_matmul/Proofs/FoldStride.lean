import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOImplements
import VeriTile.Triton.Float.PerAddressProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.Int4Matmul.FoldStride
open VeriTile Triton FP.Structural Safety KernelSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def originalBody (p : Parameters) : List ComputeStmt :=
  match (KernelSource.kernel p).surfaceBody[17]? with
  | some (ComputeStmt.forRange _ _ _ _ body) => body
  | _ => []
def candidateBody (p : Parameters) : List ComputeStmt :=
  match (KernelSource.foldStrideKernel p).surfaceBody[17]? with
  | some (ComputeStmt.forRange _ _ _ _ body) => body
  | _ => []
def originalHead (p : Parameters) := (originalBody p).take 5
def candidateHead (p : Parameters) := (candidateBody p).take 5

theorem scalar_stride (v : Values α .nat []) (a b : Nat) :
    bop (fun x y : Nat=>x*y) .nil (bop (fun x y : Nat=>x*y) .nil v (fun _=>a)) (fun _=>b) =
      bop (fun x y : Nat=>x*y) .nil v (fun _=>a*b) := by
  funext i
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_assoc]

theorem head_run (M : Algebra α) (p : Parameters) (s : FP.Structural.State α) :
    run M (originalHead p) s = run M (candidateHead p) s := by
  cases hk : s.regs .nat [] "k" <;>
    cases ho : s.regs .nat [p.blockInner] "offs_k" <;>
    cases hn : s.regs .nat [p.blockColumns] "offs_bn"
  all_goals simp [originalHead,candidateHead,originalBody,candidateBody,KernelSource.kernel,KernelSource.foldStrideKernel,
    ComputeKernel.surfaceBody,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,Option.bind,numeric,bop,natLt,numericLt,numericEq,
    integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    hk,ho,hn,Nat.mul_assoc]
  simp only [scalar_stride]

theorem pure_safe (M : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (h : ∀ st∈code,∀ s : FP.Structural.State α,stepSafe M bounds st s) (s : FP.Structural.State α) :
    traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply (traceSafeList_cons M bounds st rest s).mpr
    exact ⟨h st (by simp) s,fun next _=>ih (fun stmt hs=>h stmt (by simp [hs])) next⟩

theorem original_head_safe (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : FP.Structural.State α) :
    traceSafeList M bounds (originalHead p) s := by
  apply pure_safe
  simp [originalHead,originalBody,KernelSource.kernel,ComputeKernel.surfaceBody]
  all_goals simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem candidate_head_safe (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : FP.Structural.State α) :
    traceSafeList M bounds (candidateHead p) s := by
  apply pure_safe
  simp [candidateHead,candidateBody,KernelSource.foldStrideKernel,ComputeKernel.surfaceBody]
  all_goals simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem body_run (M : Algebra α) (p : Parameters) (s : FP.Structural.State α) :
    run M (originalBody p) s = run M (candidateBody p) s := by
  have left : originalBody p=originalHead p ++ (originalBody p).drop 5 := rfl
  have right : candidateBody p=candidateHead p ++ (originalBody p).drop 5 := rfl
  rw [left,right,run_append,run_append,head_run]

theorem body_safe (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : FP.Structural.State α) :
    traceSafeList M bounds (originalBody p) s ↔ traceSafeList M bounds (candidateBody p) s := by
  have left : originalBody p=originalHead p ++ (originalBody p).drop 5 := rfl
  have right : candidateBody p=candidateHead p ++ (originalBody p).drop 5 := rfl
  rw [left,right,traceSafeList_append,traceSafeList_append]
  simp only [original_head_safe,candidate_head_safe,true_and,head_run]

theorem loop_run (M : Algebra α) (p : Parameters) (k : Nat) (s : FP.Structural.State α) :
    range M "k" k (steps p) 1 (originalBody p) s =
      range M "k" k (steps p) 1 (candidateBody p) s := by
  conv_lhs => rw [range]
  conv_rhs => rw [range]
  by_cases live : k<steps p
  · simp only [show (1:Nat)≠0 by decide,if_false,live,if_true,body_run]
    cases ran : run M (candidateBody p) (s.setReg "k" .nat [] (fun _=>k)) with
    | none => rfl
    | some next => exact loop_run M p (k+1) next
  · simp [live]
termination_by steps p-k

theorem loop_safe (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (k : Nat) (s : FP.Structural.State α) :
    rangeSafe M bounds "k" k (steps p) 1 (originalBody p) s ↔
      rangeSafe M bounds "k" k (steps p) 1 (candidateBody p) s := by
  conv_lhs => rw [rangeSafe_unfold]
  conv_rhs => rw [rangeSafe_unfold]
  by_cases live : k<steps p
  · simp only [show (1:Nat)≠0 by decide,if_false,live,if_true,body_safe,body_run]
    cases ran : run M (candidateBody p) (s.setReg "k" .nat [] (fun _=>k)) with
    | none => rfl
    | some next => exact and_congr_right (fun _=>loop_safe M p bounds (k+1) next)
  · simp [live]
termination_by steps p-k

theorem execution (M : Algebra α) (p : Parameters) (s : FP.Structural.State α) :
    exec M (KernelSource.kernel p) s = exec M (KernelSource.foldStrideKernel p) s := by
  have left : (KernelSource.kernel p).surfaceBody=(KernelSource.kernel p).surfaceBody.take 17 ++
      .forRange "k" 0 (steps p) 1 (originalBody p) :: (KernelSource.kernel p).surfaceBody.drop 18 := rfl
  have right : (KernelSource.foldStrideKernel p).surfaceBody=(KernelSource.kernel p).surfaceBody.take 17 ++
      .forRange "k" 0 (steps p) 1 (candidateBody p) :: (KernelSource.kernel p).surfaceBody.drop 18 := rfl
  change run M (KernelSource.kernel p).surfaceBody s=run M (KernelSource.foldStrideKernel p).surfaceBody s
  rw [left,right,run_append,run_append]
  cases run M ((KernelSource.kernel p).surfaceBody.take 17) s with
  | none => rfl
  | some next => simp only [Option.bind_some,run,FP.Structural.step,loop_run]

theorem safety (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : FP.Structural.State α) :
    TraceSafe M bounds (KernelSource.kernel p) s ↔ TraceSafe M bounds (KernelSource.foldStrideKernel p) s := by
  have left : (KernelSource.kernel p).surfaceBody=(KernelSource.kernel p).surfaceBody.take 17 ++
      .forRange "k" 0 (steps p) 1 (originalBody p) :: (KernelSource.kernel p).surfaceBody.drop 18 := rfl
  have right : (KernelSource.foldStrideKernel p).surfaceBody=(KernelSource.kernel p).surfaceBody.take 17 ++
      .forRange "k" 0 (steps p) 1 (candidateBody p) :: (KernelSource.kernel p).surfaceBody.drop 18 := rfl
  unfold TraceSafe
  rw [left,right,traceSafeList_append,traceSafeList_append]
  simp only [traceSafeList_cons,stepSafe_unfold,FP.Structural.step,loop_run,loop_safe]

#axiomsClean execution
#axiomsClean safety
end
end VeriTile.Bench.Repairs.Int4Matmul.FoldStride

import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOLoopSafety

namespace VeriTile.Bench.Repairs.MixedSparseAttention.IOSourceSafety
open VeriTile Triton FP.Structural
open Whole002 Source
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Proof-only abstraction of the existing conditional body. Specializing it
back to activeBody is definitionally the existing named source decomposition. -/
def withBody (c : Config) (body : List ComputeStmt) : ComputeKernel :=
  match parsedKernel c with
  | .mk inputs outputs statements=>.mk inputs outputs
    (statements.mapIdx (fun i st=>match i,st with
      | 3,.ifThen cond _=>.ifThen cond body
      | _,_=>st))

theorem guard_safe (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (capacity : RegionName → Nat) (body : List ComputeStmt)
    (safe : Safety.traceSafeList (backend M D) capacity
      ((parsedKernel c).surfaceBody.take 3 ++ body) s) :
    TraceSafe (backend M D) capacity (withBody c body) s := by
  rcases c with ⟨storage,q,k,v,out,seq,blocks,offsets,counts,columns,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svn,svk,soz,soh,som,sok,Z,H,T,NR,NS,NV,BM,BN,BD,scale⟩
  cases storage
  all_goals
    simp [TraceSafe,withBody,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,
      ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,State.setReg,
      ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,
      integralFloorDiv,integralMod,Region.cast] at safe ⊢
  all_goals
    refine ⟨safe.1,?_⟩
    exact fun _=>safe.2

theorem kernel_safe (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (hbn : 0<c.BN) (capacity : RegionName → Nat) (hb : Bounds c s capacity) :
    TraceSafe (backend M D) capacity (kernel c) s := by
  have safe:=guard_safe M D c s capacity (activeBody c) (active_body_safe M D c s hbn capacity hb)
  rw [kernel_named]
  exact safe

#axiomsClean kernel_safe
end
end VeriTile.Bench.Repairs.MixedSparseAttention.IOSourceSafety

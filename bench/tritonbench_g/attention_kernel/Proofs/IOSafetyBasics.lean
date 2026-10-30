import bench.tritonbench_g.attention_kernel.Proofs.RawKernel
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.AttentionKernel.IOSafety
open VeriTile Triton FP.Structural
open VeriTile.Triton.FP.ProfiledRewrite (engine)
open Whole001 Source Whole Machine
open FP.Structural.Safety
noncomputable section
variable {α : Type} [Inhabited α]

/-- Actual active source windows in one launch, derived from logical IO bounds. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  query : ∀ row<p.length,∀ d<p.dim,base p s p.q+row*p.q.row+d*p.q.dim<bounds p.Q
  key : ∀ row<p.length,∀ d<p.dim,base p s p.k+row*p.k.row+d*p.k.dim<bounds p.K
  value : ∀ row<p.length,∀ d<p.dim,base p s p.v+row*p.v.row+d*p.v.dim<bounds p.V
  bias : ∀ row<p.length,∀ c<128,base p s p.bias+row*p.bias.row+c<bounds p.Bias
  output : ∀ row<p.length,∀ d<p.dim,base p s p.o+row*p.o.row+d*p.o.dim<bounds p.Out

theorem safeBlockPtr2 (M : Algebra α) (p : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (r : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    safeOp M p bounds s (.makeBlockPtrDynOffsets r base parent block strides [a,b]) ↔
      safeOp M p bounds s base ∧ safeOp M p bounds s a ∧ safeOp M p bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem safeAdvance2 (M : Algebra α) (p : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (shape : TileShape)
    (pointer : Op .blockPtr shape) (a b : Op .int []) :
    safeOp M p bounds s (.advanceBlockPtrDyn pointer [a,b]) ↔
      safeOp M p bounds s pointer ∧ safeOp M p bounds s a ∧ safeOp M p bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (hs : ∀ st ∈ code,∀ s : State α,stepSafe M bounds st s)
    (s : State α) : traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil=>exact traceSafeList_nil _ _ _
  | cons st rest ih=>
    apply traceSafeList_cons_intro (hs st (by simp) s)
    intro t _
    exact ih (fun st h=>hs st (by simp [h])) t

end
end VeriTile.Bench.Repairs.AttentionKernel.IOSafety

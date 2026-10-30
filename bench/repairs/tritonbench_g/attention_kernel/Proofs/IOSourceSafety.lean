import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOEntrySafety
import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOBodySafety
import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOPostSafety

namespace VeriTile.Bench.Repairs.AttentionKernel.IOSafety
open VeriTile Triton FP.Structural
open VeriTile.Triton.FP.ProfiledRewrite (engine)
open Whole001 Source Whole Machine
open FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

theorem source_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (guard : Guard p) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    TraceSafe (engine (withDot M dot)) bounds (kernel p) s := by
  let entry:=Source.entry M dot p s
  obtain ⟨mid,ran,row,ctx⟩:=loopExec M dot p s entry.val guard.bn (entry_context M dot p s)
  have loopRan : step (engine (withDot M dot)) (sourceLoop p) entry.val=some mid := by
    simpa [sourceLoop,step,evalOp_unfold,(entry_bounds M dot p s).1,
      (entry_bounds M dot p s).2,entry] using ran
  have loopSafe : stepSafe (engine (withDot M dot)) bounds (sourceLoop p) entry.val := by
    simp only [sourceLoop,stepSafe_unfold,safeOp_unfold]
    simp only [evalOp_unfold,entry,(entry_bounds M dot p s).1,(entry_bounds M dot p s).2]
    exact ⟨trivial,trivial,trivial,range_safe M dot p s entry.val 0 _
      (entry_context M dot p s) guard bounds hb⟩
  change traceSafeList _ _ (kernel p).surfaceBody _
  rw [source_body,List.append_assoc,traceSafeList_append]
  refine ⟨entry_safe M dot p s bounds hb,?_⟩
  intro t ht
  have eq:=Option.some.inj (ht.symm.trans entry.property)
  subst t
  change traceSafeList _ _ (sourceLoop p::(parts p).after) _
  exact traceSafeList_cons_of_step loopSafe loopRan (post_safe M dot p s mid row _ ctx bounds hb)

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionKernel.IOSafety

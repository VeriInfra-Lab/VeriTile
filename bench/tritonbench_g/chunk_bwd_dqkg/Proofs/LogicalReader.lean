import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealIOResults
import bench.tritonbench_g.chunk_bwd_dqkg.IO

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Logical
open VeriTile Triton Whole
open scoped VeriTile.Triton.LogicalIO
noncomputable section

def formula (c : Config) (xs : Inputs c) : Outputs c where
  dq := fun p => Spec.dq c xs.tensors p.1.val (p.2.1.val/64) (p.2.1.val%64) p.2.2.val
  dk := fun p => Spec.dk c xs.tensors p.1.val (p.2.1.val/64) (p.2.1.val%64) p.2.2.val
  dg := fun p => Spec.dg c xs.tensors p.1.val p.2.1.val (p.2.2.val/64) (p.2.2.val%64)

theorem logical_iff (c : Config) (order : List (CTA c)) :
    (io c order ⊨ formula c) ↔ (IO.realIO c order).Implements (Target.mathematical c) := by
  change (IO.realIO c order).Implements
    (fun vals => (outputsEquiv c).symm (formula c (inputsEquiv c vals))) ↔ _
  have eq : (fun vals => (outputsEquiv c).symm (formula c (inputsEquiv c vals))) = Target.mathematical c := by
    funext vals o j
    fin_cases o <;> simp only [formula,Inputs.tensors,Equiv.symm_apply_apply]
    all_goals rfl
  exact (congrArg (fun f => (IO.realIO c order).Implements f) eq).to_iff

theorem logical_implements (c : Config) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    io c order ⊨ formula c :=
  (logical_iff c order).mpr (RealIOResults.implements c order ho)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Logical

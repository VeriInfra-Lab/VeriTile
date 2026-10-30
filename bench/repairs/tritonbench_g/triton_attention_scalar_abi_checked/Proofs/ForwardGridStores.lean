import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.GridAddresses
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem forwardStep_otherOut (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (forwardGrid c) pid) (i : Coord c) (hne : pid≠forwardProgram c i) :
    forwardStep P c scale pid mem "Out" (offset c i)=mem "Out" (offset c i) := by
  unfold forwardStep forwardWrite
  rw [maskedScatter_frame _ _ _ _ _ _ _ _ (Or.inr (by intro lane hl hm;exact forward_other_offset c scale pid mem hp i hne lane))]
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl (by simp [forwardParameters]))]
  exact State.scatter_frame _ _ _ _ _ _ (Or.inl (by simp [forwardParameters])) _

theorem forwardStep_otherRow (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (forwardGrid c) pid) (i : Coord c) (hne : pid≠forwardProgram c i)
    (r : RegionName) (hr : r="L" ∨ r="M") :
    forwardStep P c scale pid mem r (rowOffset c i.1 i.2.1)=mem r (rowOffset c i.1 i.2.1) := by
  unfold forwardStep forwardWrite
  rw [maskedScatter_frame _ _ _ _ _ _ _ _ (Or.inl (by rcases hr with rfl|rfl <;> simp [forwardParameters]))]
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inr (by intro lane hl;exact forward_other_row c scale pid mem hp i hne lane))]
  exact State.scatter_frame _ _ _ _ _ _ (Or.inr (by intro lane hl;exact forward_other_row c scale pid mem hp i hne lane)) _

#axiomsClean forwardStep_otherOut
#axiomsClean forwardStep_otherRow
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid

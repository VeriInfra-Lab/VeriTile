import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.FPReadback
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_Linearization
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPWorkset
open VeriTile.Triton FP.Structural VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
variable {α : Type} [Inhabited α]

def initial (h : Call) (data : Data α) (b : Fin h.batches) (n : Fin h.N) : α :=
  data.output (outputOffset h b.val n.val)

def step (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (b : Fin h.batches) (n : Fin h.N) (split : Fin 64) (old : α) : α :=
  P.atomic h.outputType old (payload M P h data b.val (data.indices b.val).toNat
    split.val ⟨n.val,n.isLt.trans_le h.rowsCovered⟩)

abbrev Run (M : Algebra α) (P : Primitives α) (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (data : Data α) := VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Linearization.Run h orders data.indices (initial h data) (step M P h data)

/-- Each local runtime transition is the observable atomic update of a complete
source execution, even after arbitrary changes of all output cells. This is
proved from the source; it is not a runtime or numerical premise. -/
theorem source_event (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (b : Fin h.batches) (n : Fin h.N) (split : Fin 64)
    (valid : 0 ≤ data.indices b.val) (active : data.indices b.val ≠ -1)
    (current : Nat → α) :
    ∃ t,exec (machine M P) (Surface.source h)
      (entry h b.val split.val {data with output := current})=some t ∧
      t.mem "Y" (outputOffset h b.val n.val)=
        .mk h.outputType.toTileDType (ofFloat _ (step M P h data b n split
          (current (outputOffset h b.val n.val)))) ∧
      (∀ (r : RegionName) (off : Nat),¬(r="Y" ∧ ∃ j : Fin h.N,
        off=outputOffset h b.val j.val) →
        t.mem r off=(entry h b.val split.val {data with output := current}).mem r off) := by
  obtain ⟨t,execution,_,readback,frame⟩ := FPReadback.every_program M P h ⟨b,split⟩
    {data with output := current} (Or.inr valid)
  refine ⟨t,execution,?_,?_⟩
  · have result := readback active n
    have payloadEq := FPReadback.payload_independent M P h {data with output := current}
      data rfl rfl rfl b.val (data.indices b.val).toNat split.val
    simpa only [step,payloadEq] using result
  · intro r off outside
    apply frame
    exact fun hit => outside hit.2

theorem readback (M : Algebra α) (P : Primitives α) (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (data : Data α) (run : Run M P h orders data) (b : Fin h.batches) (n : Fin h.N) :
    run.output b n=target M P h orders data b n := by
  exact VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Linearization.Run.readback h orders data.indices (initial h data) (step M P h data) run b n

/-- Every active event's transition is realized by the complete typed source,
at arbitrary current output values, with the complete source memory frame. -/
def SourceWorkset (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) : Prop :=
  ∀ (b : Fin h.batches) (n : Fin h.N) (split : Fin 64),data.indices b.val ≠ -1 →
  ∀ current : Nat → α,∃ t,exec (machine M P) (Surface.source h)
    (entry h b.val split.val {data with output := current})=some t ∧
    t.mem "Y" (outputOffset h b.val n.val)=
      .mk h.outputType.toTileDType (ofFloat _ (step M P h data b n split
        (current (outputOffset h b.val n.val)))) ∧
    (∀ (r : RegionName) (off : Nat),¬(r="Y" ∧ ∃ j : Fin h.N,
      off=outputOffset h b.val j.val) →
      t.mem r off=(entry h b.val split.val {data with output := current}).mem r off)

theorem source_workset (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (valid : ∀ b,b < h.batches → data.indices b=-1 ∨ 0 ≤ data.indices b) :
    SourceWorkset M P h data := by
  intro b n split active current
  exact source_event M P h data b n split ((valid b.val b.isLt).resolve_left active) active current

#axiomsClean source_event
#axiomsClean readback
#axiomsClean source_workset
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPWorkset

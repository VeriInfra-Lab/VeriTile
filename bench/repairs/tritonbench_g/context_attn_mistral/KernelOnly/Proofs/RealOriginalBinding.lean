import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.RawRealResults
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.FormulaAgreement

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section

def snapshot (c : Host.Call) (capacity : RegionBounds) (s : BlockState) : Inputs c capacity :=
  fun port i=>(inputType port).read s (inputRegion c port) i.val

theorem snapshot_binding (c : Host.Call) (capacity : RegionBounds) (s : BlockState) :
    (programIO c capacity).InputBinding s (snapshot c capacity s) := by
  intro port i on
  rfl

/-- Every old admitted flat input supplies the new finite-array metadata and
geometry, with no additional input restriction. -/
theorem original_admitted (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (valid : Host.Valid c A s) :
    Geometry c A.extent ∧ Metadata c A.extent (snapshot c A.extent s) := by
  have agree:=input_agreement c A.extent (snapshot c A.extent s) s (snapshot_binding c A.extent s)
  refine ⟨geometry_from_original c A s valid,?_,?_⟩
  · intro b hb
    exact (agree.start b (lt_of_lt_of_le hb valid.start_bound)).symm.trans (valid.start_value b hb)
  · intro b hb
    exact (agree.sequence b (lt_of_lt_of_le hb valid.length_bound)).symm.trans (valid.length_value b hb)

/-- The public array formula is exactly the previously accepted logical Tensor
formula on every active program/lane, including an intentionally partial grid. -/
theorem original_formula (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (input : Logical.Tensors) (valid : Host.Valid c A s) (reads : Logical.Resolved c s input)
    (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) (on : c.active idx i) :
    formula c A.extent (snapshot c A.extent s) 0 (lanes c (idx,i))=
      realTarget c input (c.launchAt idx).batch (c.launchAt idx).head
        (Formula.queryRow c.layout (c.launchAt idx) i.1.val) i.2.1.val := by
  simp only [formula,Equiv.symm_apply_apply]
  have agreement:=formula_agreement c A.extent (Program.inputs s c.regions)
    (logicalInputs c A.extent (snapshot c A.extent s))
    (input_agreement c A.extent _ s (snapshot_binding c A.extent s))
    (c.launchAt idx) (Host.program_bounds c A s valid idx) (Host.scale c.layout.D) i.1 i.2.1
  rw [←agreement]
  have lane : i.1.val<128 := by simpa only [c.blockM] using i.1.isLt
  have quotient : (Formula.queryRow c.layout (c.launchAt idx) i.1.val)/128=(c.launchAt idx).block := by
    simp only [Formula.queryRow,c.blockM]
    omega
  have remainder : (Formula.queryRow c.layout (c.launchAt idx) i.1.val)%128=i.1.val := by
    simp only [Formula.queryRow,c.blockM]
    omega
  have independent:=Logical.expected_eq c A s input valid reads
    (c.launchAt idx).batch (c.launchAt idx).head
    (Formula.queryRow c.layout (c.launchAt idx) i.1.val) i.2.1.val
    (Host.batch_bound c idx) (Host.head_bound c idx) on i.2.1.isLt
  simpa only [Host.Call.logicalExpected,quotient,remainder] using independent

#axiomsClean snapshot_binding
#axiomsClean original_admitted
#axiomsClean original_formula
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface

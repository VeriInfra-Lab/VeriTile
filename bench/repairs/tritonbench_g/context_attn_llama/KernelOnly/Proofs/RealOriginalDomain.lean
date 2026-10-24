import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.RealBindings

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section

def snapshot (c : Host.Call) (capacity : RegionBounds) (s : BlockState) : Inputs c capacity :=
  fun port off=>(inputType port).read s (inputRegion c port) off.val

theorem snapshot_binding (c : Host.Call) (capacity : RegionBounds) (s : BlockState) :
    (programIO c capacity).InputBinding s (snapshot c capacity s) := fun _ _ _=>rfl

/-- Every old admissible raw state supplies the new finite logical inputs.
No payload/tag guard or additional metadata restriction is introduced. -/
theorem metadata_from_original (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (h : Host.Valid c A s) : Metadata c A.extent (snapshot c A.extent s) := by
  have agree:=input_agreement c A.extent (snapshot c A.extent s) s (snapshot_binding c A.extent s)
  exact {
    start:=fun b hb=>(agree.start b (lt_of_lt_of_le hb h.start_bound)).symm.trans (h.start_value b hb)
    sequence:=fun b hb=>(agree.sequence b (lt_of_lt_of_le hb h.length_bound)).symm.trans (h.length_value b hb)
    prompt:=fun b hb=>(agree.prompt b (lt_of_lt_of_le hb h.prompt_bound)).symm.trans (h.prompt_value b hb)
    request:=fun b hb=>(agree.request b (lt_of_lt_of_le hb h.request_bound)).symm.trans (h.request_value b hb)
    pages:=fun b hb j hj=>(agree.pages _ (h.page_bound b hb j hj)).symm.trans (h.page_value b hb j hj)
  }

#axiomsClean snapshot_binding
#axiomsClean metadata_from_original
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface

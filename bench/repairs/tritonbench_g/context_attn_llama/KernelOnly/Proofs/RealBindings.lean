import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 8000

/-- Extents-only adapter to reuse the established layout lemmas. No flat
execution or allocator behavior is introduced by this helper. -/
def capacityAlloc (capacity : RegionBounds) : FlatAlloc := ⟨"",[],fun _=>0,capacity⟩

theorem input_read (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity)
    (s : BlockState) (bound : (programIO c capacity).InputBinding s values)
    (i : Fin 8) (off : Nat) (h : off<capacity (inputRegion c i)) :
    (inputType i).read s (inputRegion c i) off=values i ⟨off,h⟩ :=
  bound i ⟨off,h⟩ trivial

structure Agreement (c : Host.Call) (capacity : RegionBounds) (a b : Formula.Inputs) : Prop where
  q : ∀ off,off<capacity c.regions.query → a.q off=b.q off
  k : ∀ off,off<capacity c.regions.key → a.k off=b.k off
  v : ∀ off,off<capacity c.regions.value → a.v off=b.v off
  start : ∀ off,off<capacity c.regions.start → a.start off=b.start off
  sequence : ∀ off,off<capacity c.regions.sequence → a.sequence off=b.sequence off
  prompt : ∀ off,off<capacity c.regions.prompt → a.prompt off=b.prompt off
  request : ∀ off,off<capacity c.regions.request → a.request off=b.request off
  pages : ∀ off,off<capacity c.regions.pages → a.pages off=b.pages off

theorem input_agreement (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity)
    (s : BlockState) (bound : (programIO c capacity).InputBinding s values) :
    Agreement c capacity (Program.inputs s c.regions) (logicalInputs c capacity values) := by
  constructor
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 0 off h
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 1 off h
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 2 off h
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 3 off h
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 4 off h
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 5 off h
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 6 off h
  · intro off h
    simpa [Program.inputs,logicalInputs,array,inputRegion,inputType,ChanTy.read,h] using input_read c capacity values s bound 7 off h

theorem host_valid (c : Host.Call) (capacity : RegionBounds) (geometry : Geometry c capacity)
    (values : Inputs c capacity) (metadata : Metadata c capacity values)
    (s : BlockState) (bound : (programIO c capacity).InputBinding s values) :
    Host.Valid c (capacityAlloc capacity) s := by
  have agree:=input_agreement c capacity values s bound
  refine {
    start_value:=fun b hb=>(agree.start b (lt_of_lt_of_le hb geometry.start_bound)).trans (metadata.start b hb)
    length_value:=fun b hb=>(agree.sequence b (lt_of_lt_of_le hb geometry.length_bound)).trans (metadata.sequence b hb)
    prompt_value:=fun b hb=>(agree.prompt b (lt_of_lt_of_le hb geometry.prompt_bound)).trans (metadata.prompt b hb)
    request_value:=fun b hb=>(agree.request b (lt_of_lt_of_le hb geometry.request_bound)).trans (metadata.request b hb)
    page_value:=fun b hb j hj=>(agree.pages _ (geometry.page_bound b hb j hj)).trans (metadata.pages b hb j hj)
    prompt_bound:=geometry.prompt_bound
    request_bound:=geometry.request_bound
    page_bound:=geometry.page_bound
    start_bound:=geometry.start_bound
    length_bound:=geometry.length_bound
    group_pos:=geometry.group_pos
    grouped:=geometry.grouped
    query:=geometry.query
    key:=geometry.key
    value:=geometry.value
    out:=geometry.out
    packed:=geometry.packed
    output_view:=geometry.output_view
    input_separate:=geometry.input_separate
  }

/-- Every previous raw input admits exactly these geometry and finite metadata
conditions; the format change adds no layout or value restriction. -/
theorem geometry_from_original (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (h : Host.Valid c A s) : Geometry c A.extent := by
  exact {
    prompt_bound:=h.prompt_bound
    request_bound:=h.request_bound
    page_bound:=h.page_bound
    start_bound:=h.start_bound
    length_bound:=h.length_bound
    group_pos:=h.group_pos
    grouped:=h.grouped
    query:=h.query
    key:=h.key
    value:=h.value
    out:=h.out
    packed:=h.packed
    output_view:=h.output_view
    input_separate:=h.input_separate
  }

#axiomsClean input_agreement
#axiomsClean host_valid
#axiomsClean geometry_from_original
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface

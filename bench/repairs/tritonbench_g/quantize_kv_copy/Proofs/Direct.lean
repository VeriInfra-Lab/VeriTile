import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Source
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.FP

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Direct
noncomputable section

structure Result where
  returned : Option Unit
  outputObject : Nat
  scaleObject : Nat
  outputMetadata : VeriTile.Torch.TensorMetadata
  scaleMetadata : VeriTile.Torch.TensorMetadata
  heap : Heap

/-- The checked call preserves the actual caller-owned objects and data spans. -/
def runWrapper (actual : Operations Word Scale) (m : Inputs) (w : Nat → MetadataWord)
    (objects : Role → Nat) (plan : Plan (request m w)) (before : Heap) : Except InputError Result := do
  checkInputs m w
  let result := Native.runGrid actual (request m w) plan decodeWord scaleByte (directLayout m objects)
    (Host.ascending (request m w)).order before
  return ⟨none,objects .output,objects .scale,m .output,m .scale,result⟩

/-- Structural result for arbitrary native scalar behavior. No finite-data or
conversion-domain restriction is added to the checked input contract. -/
theorem correct (actual : Operations Word Scale) (m : Inputs) (w : Nat → MetadataWord)
    (h : InputContract m w) (objects : Role → Nat) (plan : Plan (request m w))
    (before : Heap) (binding : TensorBinding m w before) :
    ∃ out, runWrapper actual m w objects plan before=.ok out ∧ out.returned=none ∧
      out.outputObject=objects .output ∧ out.scaleObject=objects .scale ∧
      out.outputMetadata=m .output ∧ out.scaleMetadata=m .scale ∧
      (∀ s a c, s<(request m w).programs → a<(request m w).groups → c<groupWidth →
        out.heap (outputAddress (request m w) (directLayout m objects) ((request m w).destination s) a c)=
          codes actual plan decodeWord (directLayout m objects) before s a c) ∧
      (∀ s a (b : Fin 4), s<(request m w).programs → a<(request m w).groups →
        out.heap (scaleAddress (request m w) (directLayout m objects) ((request m w).destination s) a b)=
          scales actual plan decodeWord scaleByte (directLayout m objects) before s a b) ∧
      (∀ byte, ¬Touched (request m w) (directLayout m objects) byte → out.heap byte=before byte) := by
  have accepted := (checkInputs_iff m w).mpr h
  have domain := checked_domain m w h
  have alloc := direct_allocations m w h objects
  have metadata := direct_metadata m w objects before binding
  have native := Native.grid_exact m w h objects before binding actual plan decodeWord scaleByte
    (Host.ascending (request m w)).order (by
      intro s hs
      exact List.mem_range.mp ((Host.ascending (request m w)).complete.mem_iff.mp hs))
  have values := whole_readback actual (request m w) domain plan decodeWord scaleByte
    (directLayout m objects) alloc (Host.ascending (request m w)) before metadata
  refine ⟨⟨none,objects .output,objects .scale,m .output,m .scale,
    (call actual (request m w) plan decodeWord scaleByte (directLayout m objects)
      (Host.ascending (request m w)) before).1⟩,?_,rfl,rfl,rfl,rfl,rfl,values.1,values.2,?_⟩
  · simp [runWrapper,accepted,native,call]
  · exact whole_frame actual (request m w) domain plan decodeWord scaleByte (directLayout m objects)
      alloc (Host.ascending (request m w)) before metadata

#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.Direct

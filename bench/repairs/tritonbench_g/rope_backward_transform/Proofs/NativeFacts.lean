import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Native
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem span_bound (rows half rs cs time j : Nat) (ht : time<rows) (hj : j<half) :
    time*rs+j*cs+1≤span rows half rs cs := by
  have h1 := Nat.mul_le_mul_right rs (show time≤rows-1 by omega)
  have h2 := Nat.mul_le_mul_right cs (show j≤half-1 by omega)
  simp only [span,show ¬(rows=0∨half=0) from by omega,↓reduceIte]
  omega

theorem output_bounded {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α)
    (r : Request) (m : Source.Memory α) (call : Preparation api embed r m)
    (key : Bool) (a h c : Nat) (ha : a<r.rows) (hh : h<r.heads key) (hc : c<r.dimension) :
    (call.output key).base≤(call.output key).base+4*r.offset key a h c ∧
      (call.output key).base+4*r.offset key a h c+4≤(call.output key).allocationEnd ∧
      (call.output key).allocationEnd≤2^64 := by
  have ho := offset_bound r key a h c ha hh hc
  have hf := call.outputFits key
  omega

theorem copied_input_frame {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Preparation api embed r m)
    (key : Bool) (copied : call.reused key=Bool.false) (b : Nat)
    (hlo : (call.input key).base≤b) (hhi : b<(call.input key).base+4*r.size key) :
    api.byte (entry api embed project op r m call).memory b=api.byte call.before b := by
  change api.byte (Memory.run api project op r m _ _ _) b=_
  rw [Memory.byte_frame api project op r m _ _ _ b]
  · exact call.frame b (call.inputOwned key b hlo hhi)
  · intro l hl hal
    have hr := launchRows_valid r l.row ((Memory.mem_locations r _ l).mp hl).1
    have ho := offset_bound r l.key l.row l.head (l.column r) hr hal.1 (Memory.column_bound r l hal)
    have sep := call.copiedInputSeparate key l.key copied
    dsimp only
    omega

/-- Physical reads of the next row are unchanged by all completed other rows. -/
theorem current_gradient {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Preparation api embed r m)
    (xs : List Nat) (valid : ∀ a∈xs,a<r.rows) (key : Bool) (a h c : Nat)
    (ha : a<r.rows) (hh : h<r.heads key) (hc : c<r.dimension) (fresh : a∉xs) :
    embed (api.read (Memory.run api project op r m (fun k => (call.output k).base) xs call.packed)
      ((call.output key).base+4*r.offset key a h c))=(m (region key) (r.offset key a h c)).read .fp32 := by
  rw [Memory.coordinate_frame api project op r m _ xs call.packed call.outputSeparate valid
    key a h c ha hh hc (Or.inl fresh),call.packedBinding key a h c ha hh hc,call.inputBinding key a h c ha hh hc]

theorem current_coefficient {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Preparation api embed r m)
    (xs : List Nat) (valid : ∀ a∈xs,a<r.rows) (time j : Nat) (ht : time<r.sequence) (hj : j<r.half) :
    embed (api.read (Memory.run api project op r m (fun k => (call.output k).base) xs call.packed)
      (tableAddress call.cosine time j))=(m "C" (time*r.cosRowStride+j*r.cosColStride)).read .fp32 ∧
    embed (api.read (Memory.run api project op r m (fun k => (call.output k).base) xs call.packed)
      (tableAddress call.sine time j))=(m "S" (time*r.sinRowStride+j*r.sinColStride)).read .fp32 := by
  have hc := span_bound r.cosRows r.half r.cosRowStride r.cosColStride time j (lt_of_lt_of_le ht call.tableRows.1) hj
  have hs := span_bound r.sinRows r.half r.sinRowStride r.sinColStride time j (lt_of_lt_of_le ht call.tableRows.2) hj
  have cb : ∀ k,tableAddress call.cosine time j+4≤(call.output k).base ∨
      (call.output k).base+4*r.size k≤tableAddress call.cosine time j := by
    intro k
    have sep := (call.coefficientSeparate k).1
    unfold tableAddress
    rw [call.coefficientStrides.1]
    simp only [List.getElem!_cons_zero,List.getElem!_cons_succ]
    omega
  have sb : ∀ k,tableAddress call.sine time j+4≤(call.output k).base ∨
      (call.output k).base+4*r.size k≤tableAddress call.sine time j := by
    intro k
    have sep := (call.coefficientSeparate k).2
    unfold tableAddress
    rw [call.coefficientStrides.2]
    simp only [List.getElem!_cons_zero,List.getElem!_cons_succ]
    omega
  constructor
  · rw [Memory.read_frame api project op r m _ xs call.packed _ cb valid,
      call.preservedCosine time j ht hj,call.cosineBinding time j ht hj]
  · rw [Memory.read_frame api project op r m _ xs call.packed _ sb valid,
      call.preservedSine time j ht hj,call.sineBinding time j ht hj]

#axiomsClean output_bounded
#axiomsClean copied_input_frame
#axiomsClean current_gradient
#axiomsClean current_coefficient
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.Native

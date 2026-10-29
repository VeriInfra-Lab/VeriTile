import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Geometry
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Storage
open VeriTile.Triton Geometry
abbrev Buffer := VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.Buffer
abbrev Disjoint := VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.Disjoint
abbrev signedLimit := VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.signedLimit
abbrev signed64 := VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.signed64
abbrev widen32 := VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.widen32
abbrev extent_bound := VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.extent_bound
abbrev pointer_word := VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.pointer_word

/-- The same successful compact allocations used by the actual wrapper.
Signed tensor storage sizes and physical byte windows are ordinary allocation
facts, not tensor-dimension admission guards. Output/index widths remain local
storage facts; they are not assumed to be FP32/int32. -/
structure Placement (h : Call) where
  input : Buffer
  weights : Buffer
  output : Buffer
  indices : Buffer
  inputSize : input.extent=h.batches*h.paddedK
  weightSize : weights.extent=h.adapters*(h.BN*h.paddedK)
  outputSize : output.extent=h.batches*h.BN
  indexSize : indices.extent=h.batches
  inputWidth : input.bytes=2
  weightWidth : weights.bytes=2
  indexWidth : indices.bytes=4 ∨ indices.bytes=8
  separate : List.Pairwise Disjoint [input,weights,output,indices]

/-- Covers every generated address, including inactive n/K lanes. -/
theorem all_source_offsets (h : Call) (s : Placement h) (p : Program h)
    (slot : Fin h.adapters) (c : Fin h.cycles) (e : Fin 256) (n : Fin h.BN) :
    inputOffset h p.batch.val (rank p.split.val c.val e.val)<s.input.extent ∧
    weightOffset h slot.val n.val (rank p.split.val c.val e.val)<s.weights.extent ∧
    outputOffset h p.batch.val n.val<s.output.extent ∧ p.batch.val<s.indices.extent := by
  rw [s.inputSize,s.weightSize,s.outputSize,s.indexSize]
  have inside := rank_inside h p.split c e
  exact ⟨input_inside h p _ inside,weight_inside h slot n _ inside,output_inside h p n,p.batch.isLt⟩

/-- Source-expression i64 offset results and final byte-address calculations
are exact for all allocated lanes. Individual nonnegative partial sums and
products are bounded above by these same full offsets. -/
theorem native_addresses (h : Call) (s : Placement h) (p : Program h)
    (slot : Fin h.adapters) (c : Fin h.cycles) (e : Fin 256) (n : Fin h.BN) :
    let x := inputOffset h p.batch.val (rank p.split.val c.val e.val)
    let w := weightOffset h slot.val n.val (rank p.split.val c.val e.val)
    let y := outputOffset h p.batch.val n.val
    (BitVec.ofNat 64 x).toInt=(x:Int) ∧
    (BitVec.ofNat 64 w).toInt=(w:Int) ∧
    (BitVec.ofNat 64 y).toInt=(y:Int) ∧
    (BitVec.ofNat 64 s.input.base+BitVec.ofNat 64 x*BitVec.ofNat 64 s.input.bytes).toNat=s.input.address x ∧
    (BitVec.ofNat 64 s.weights.base+BitVec.ofNat 64 w*BitVec.ofNat 64 s.weights.bytes).toNat=s.weights.address w ∧
    (BitVec.ofNat 64 s.output.base+BitVec.ofNat 64 y*BitVec.ofNat 64 s.output.bytes).toNat=s.output.address y := by
  obtain ⟨x,w,y,_⟩ := all_source_offsets h s p slot c e n
  exact ⟨signed64 _ (x.trans (extent_bound s.input)),signed64 _ (w.trans (extent_bound s.weights)),
    signed64 _ (y.trans (extent_bound s.output)),pointer_word s.input _ x,
    pointer_word s.weights _ w,pointer_word s.output _ y⟩

theorem loop_bounds (h : Call) (s : Placement h) (p : Program h) :
    h.K<signedLimit ∧ h.cycles*16384<signedLimit ∧ h.paddedK<signedLimit := by
  have cap : h.paddedK≤s.input.extent := by
    rw [s.inputSize]
    exact Nat.le_mul_of_pos_left _ (by have := p.batch.isLt; omega)
  have bound := cap.trans_lt (extent_bound s.input)
  have covers := padded h
  have cycles := Nat.mul_le_mul_right 16384 (Nat.le_max_right 1 h.cycles)
  exact ⟨covers.trans_lt bound,cycles.trans_lt bound,bound⟩

#axiomsClean all_source_offsets
#axiomsClean native_addresses
#axiomsClean loop_bounds
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Storage

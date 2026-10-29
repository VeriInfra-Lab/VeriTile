import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Chunks

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Finite
open VeriTile.Bench.TritonBenchG.VectorAddition
noncomputable section
set_option maxHeartbeats 4000000

abbrev grid := Native.grid
abbrev mask := Native.mask
abbrev batches := F8ConversionUtils.Chunking.batches
abbrev chunkBatches := F8ConversionUtils.Chunking.chunkBatches

theorem small (n c : Nat) : chunkLength n c<2^31 := F8ConversionUtils.Chunking.chunk_small n c
theorem active (n c i : Nat) (hi : i<chunkLength n c) : chunkStart c+i<n := F8ConversionUtils.Chunking.chunk_active n c i hi
theorem coverage (n i : Nat) : i∈(batches n).flatten ↔ i<n := F8ConversionUtils.Chunking.full_coverage n i

theorem shifted (v : View) (start i : Nat) :
    (v.shift start).signedAddress i=v.signedAddress (start+i) := by
  simp only [View.shift,View.signedAddress,Nat.cast_add]
  ring
theorem shifted_address (v : View) (start i : Nat) :
    (v.shift start).address i=v.address (start+i) := congrArg Int.toNat (shifted v start i)

def pointerWord (v : View) (pid lane : Nat) : BitVec 64 :=
  BitVec.ofInt 64 v.pointer+(Native.offset pid lane).signExtend 64*BitVec.ofInt 64 v.byteStride
def pointer (v : View) (pid lane : Nat) : Nat := (pointerWord v pid lane).toNat

/-- Native element strides are cast to int64 before multiplication. Encoding
their byte stride commutes with multiplying by the exact storage width. -/
theorem element_scaling (pointer stride : Int) (width : Nat) (pid lane : Nat) :
    pointerWord ⟨pointer,stride*width⟩ pid lane=
      BitVec.ofInt 64 pointer+(Native.offset pid lane).signExtend 64*
        BitVec.ofInt 64 stride*BitVec.ofNat 64 width := by
  simp [pointerWord,BitVec.ofInt_mul,BitVec.ofInt_natCast,BitVec.mul_assoc]

theorem word_encoding (v : View) (count pid lane : Nat) (hn : count<2^31)
    (hp : pid<grid count) (hl : lane<1024) :
    pointerWord v pid lane=BitVec.ofInt 64 (v.signedAddress (pid*1024+lane)) := by
  rw [pointerWord]
  change BitVec.ofInt 64 v.pointer+
    (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane).signExtend 64*BitVec.ofInt 64 v.byteStride = _
  rw [VeriTile.Bench.TritonBenchG.VectorAddition.Finite.source_index_sign_extend count pid lane hn hp hl]
  simp only [View.signedAddress,BitVec.ofInt_add,BitVec.ofInt_mul,BitVec.ofInt_natCast]

theorem pointer_exact (v : View) (count pid lane : Nat) (hn : count<2^31)
    (hp : pid<grid count) (hl : lane<1024)
    (lower : 0≤v.signedAddress (pid*1024+lane))
    (upper : v.signedAddress (pid*1024+lane)<(2^64 : Nat)) :
    pointer v pid lane=v.address (pid*1024+lane) := by
  have encoded : v.signedAddress (pid*1024+lane)=(v.address (pid*1024+lane) : Int) := by
    simp [View.address,Int.toNat_of_nonneg lower]
  have fits : v.address (pid*1024+lane)<2^64 := by rw [encoded] at upper;exact_mod_cast upper
  rw [pointer,word_encoding v count pid lane hn hp hl,encoded,
    BitVec.ofInt_natCast,BitVec.toNat_ofNat,Nat.mod_eq_of_lt fits]

theorem checked_pointer (out : Bool) (v : ViewMetadata) (h : ViewContract out v)
    (n start count pid lane : Nat) (same : n=v.n) (hn : count<2^31)
    (hp : pid<grid count) (hl : lane<1024) (inside : start+(pid*1024+lane)<n) :
    pointer (v.view.shift start) pid lane=v.view.address (start+(pid*1024+lane)) := by
  have bounds := accessed_bounds out v h (start+(pid*1024+lane)) 0 (by omega) (width_pos v)
  rw [pointer_exact _ count pid lane hn hp hl,shifted_address]
  · rw [shifted];exact bounds.2.2.1
  · rw [shifted];simpa only [Nat.cast_zero,Int.add_zero] using bounds.2.2.2

theorem output_view (v : ViewMetadata) (h : ViewContract Bool.true v) : v.view=v.outputView := by
  simp [ViewMetadata.view,ViewMetadata.outputView,output_width v h]

theorem physical_disjoint (x y : ViewMetadata) (h : InputContract x y) :
    ∀ i,i<x.n → ∀ k,k<x.width → ∀ j,j<x.n → x.view.address i+k≠y.outputView.address j := by
  intro i hi k hk j hj equal
  have xb := (accessed_bounds Bool.false x h.input i 0 hi (width_pos x)).2.2.1
  have yb := (accessed_bounds Bool.true y h.output j 0 (by rw [←h.shape];exact hj) (width_pos y)).2.2.1
  rw [output_view y h.output] at yb
  apply h.disjoint i hi k hk j hj
  have equality := congrArg Int.ofNat equal
  change ((x.view.signedAddress i).toNat : Int)+(k : Int)=((y.outputView.signedAddress j).toNat : Int) at equality
  simpa only [Int.toNat_of_nonneg xb,Int.toNat_of_nonneg yb] using equality

theorem indices (n c pid lane : Nat) (hp : pid<grid (chunkLength n c)) (hl : lane<1024) :
    (Native.offset pid lane).toInt=(pid*1024+lane : Nat) ∧
    mask (chunkLength n c) pid lane=decide (pid*1024+lane<chunkLength n c) := F8ConversionUtils.Chunking.finite_indices n c pid lane hp hl

theorem alignment (out : Bool) (v : ViewMetadata) (h : ViewContract out v)
    (i : Nat) (hi : i<v.n) : v.view.address i % v.width=0 := by
  have positive := (accessed_bounds out v h i 0 hi (width_pos v)).2.2.1
  have original := h.alignment (by omega)
  have aligned : v.view.signedAddress i % (v.width : Int)=0 := by
    simp [View.signedAddress,ViewMetadata.view,Int.add_emod,Int.mul_emod,original]
  have equality : (v.view.address i : Int)=v.view.signedAddress i := by
    exact Int.toNat_of_nonneg positive
  rw [←equality] at aligned
  exact_mod_cast aligned

#axiomsClean checked_pointer
#axiomsClean physical_disjoint
#axiomsClean indices
#axiomsClean alignment
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel.Finite

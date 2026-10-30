import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordGrid
import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordGrid
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
local instance : Inhabited ScaledWord.Word := ⟨FP.Value.zero .fp32⟩

/-- Bounds are on element offsets from the supplied tensor's data pointer.
These are ordinary readable/writable tensor-view contracts, not result premises. -/
def ViewBounds (rows columns rowStride columnStride capacity : Nat) : Prop :=
  ∀ r c, r<rows → c<columns → r*rowStride+c*columnStride<capacity

namespace BlockWord

theorem input_access {α : Type} (p : Parameters) (s : State α) (capacity : Nat)
    (hv : ViewBounds p.rows p.inner p.inputRowStride p.inputInnerStride capacity)
    (k : Nat) (i : AIndex p)
    (hload : (aPointer p s k).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true) :
    (aPointer p s k).address (TileShape.indexToList _ i)<capacity := by
  simp only [aPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,decide_eq_true_eq] at hload
  have h := hv _ _ hload.1 hload.2
  simpa [aPointer,TileShape.indexToList,BlockPtr.address_2d_offsets,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h

theorem weight_access {α : Type} (p : Parameters) (s : State α) (capacity : Nat)
    (hv : ViewBounds p.inner p.columns p.weightInnerStride p.weightColumnStride capacity)
    (k : Nat) (i : BIndex p)
    (hload : (bPointer p s k).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true) :
    (bPointer p s k).address (TileShape.indexToList _ i)<capacity := by
  simp only [bPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,decide_eq_true_eq] at hload
  have h := hv _ _ hload.1 hload.2
  simpa [bPointer,TileShape.indexToList,BlockPtr.address_2d_offsets,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h

theorem output_access {α : Type} (p : Parameters) (s : State α) (capacity : Nat)
    (hv : ViewBounds p.rows p.columns p.outputRowStride p.outputColumnStride capacity)
    (i : CIndex p) (hi : active p s i) : outputOffset p s i<capacity := by
  simpa [outputOffset,Nat.mul_comm] using hv _ _ hi.1 hi.2

end BlockWord
namespace ScaledWord

theorem loaded_inner (p : Parameters) (he : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (k : Nat) (hk : k<p.steps) (t : Fin p.blockInner)
    (hm : p.even=Bool.true ∨ t.val<p.inner-k*p.blockInner) : k*p.blockInner+t.val<p.inner := by
  rcases hm with hm|hm
  · exact even_lane_in_bounds p he hm k hk t
  · omega

theorem input_access (p : Parameters) (s : State Word) (capacity : Nat) (hM : 0<p.rows)
    (hv : ViewBounds p.rows p.inner p.inputRowStride p.inputInnerStride capacity)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (k : Nat) (hk : k<p.steps) (i : AIndex p)
    (hm : p.even=Bool.true ∨ inputInner p s i.2.1<p.inner-k*p.blockInner) : aOffset p s k i<capacity := by
  have hr : inputRow p s i.1<p.rows := Nat.mod_lt _ hM
  have hi := loaded_inner p he k hk i.2.1 hm
  have h := hv _ _ hr hi
  simpa [aOffset,inputInner,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h

theorem weight_access (p : Parameters) (s : State Word) (capacity : Nat) (hN : 0<p.columns)
    (hv : ViewBounds p.inner p.columns p.weightInnerStride p.weightColumnStride capacity)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (k : Nat) (hk : k<p.steps) (i : BIndex p)
    (hm : p.even=Bool.true ∨ inputInner p s i.1<p.inner-k*p.blockInner) : bOffset p s k i<capacity := by
  have hc : inputColumn p s i.2.1<p.columns := Nat.mod_lt _ hN
  have hi := loaded_inner p he k hk i.1 hm
  have h := hv _ _ hi hc
  simpa [bOffset,inputInner,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h

theorem output_access (p : Parameters) (s : State Word) (capacity : Nat)
    (hv : ViewBounds p.rows p.columns p.outputRowStride p.outputColumnStride capacity)
    (i : CIndex p) (hi : active p s i) : outputOffset p s i<capacity := by
  simpa [outputOffset,Nat.add_comm] using hv _ _ hi.1 hi.2

theorem scale_access (p : Parameters) (s : State Word) (capacity : Nat)
    (hv : ViewBounds p.rows 1 p.scaleRowStride p.scaleColumnStride capacity)
    (i : CIndex p) (hi : active p s i) : outputRow p s i*p.scaleRowStride<capacity := by
  simpa using hv _ 0 hi.1 (by decide)

end ScaledWord
#axiomsClean BlockWord.input_access
#axiomsClean BlockWord.weight_access
#axiomsClean BlockWord.output_access
#axiomsClean ScaledWord.loaded_inner
#axiomsClean ScaledWord.input_access
#axiomsClean ScaledWord.weight_access
#axiomsClean ScaledWord.output_access
#axiomsClean ScaledWord.scale_access
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001

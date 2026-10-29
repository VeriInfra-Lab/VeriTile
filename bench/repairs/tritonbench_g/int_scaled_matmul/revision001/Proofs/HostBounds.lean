import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Host
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Host
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem input_bounds (x : Arguments) (h : Valid x) :
    ViewBounds (blockParameters x).rows (blockParameters x).inner
      (blockParameters x).inputRowStride (blockParameters x).inputInnerStride x.a.capacity := h.aBounds

theorem weight_bounds (x : Arguments) (h : Valid x) :
    ViewBounds (blockParameters x).inner (blockParameters x).columns
      (blockParameters x).weightInnerStride (blockParameters x).weightColumnStride x.b.capacity := by
  simpa only [blockParameters,tensorBounds,h.shape.1] using h.bBounds

theorem output_bounds (x : Arguments) (h : Valid x) :
    ViewBounds (blockParameters x).rows (blockParameters x).columns
      (blockParameters x).outputRowStride (blockParameters x).outputColumnStride x.c.capacity := by
  simpa only [blockParameters,tensorBounds,h.shape.2.1,h.shape.2.2] using h.cBounds

theorem scale_bounds (x : Arguments) (scale : Tensor) (h : ScaleValid x scale) :
    ViewBounds (scaledParameters x scale).rows 1
      (scaledParameters x scale).scaleRowStride (scaledParameters x scale).scaleColumnStride scale.capacity := by
  simpa only [scaledParameters,tensorBounds,h.shape.1,h.shape.2] using h.bounds

theorem block_accesses (x : Arguments) (h : Valid x) (s : State Word) :
    (∀ k (i : BlockWord.AIndex (blockParameters x)),
      (BlockWord.aPointer (blockParameters x) s k).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true →
      (BlockWord.aPointer (blockParameters x) s k).address (TileShape.indexToList _ i)<x.a.capacity) ∧
    (∀ k (i : BlockWord.BIndex (blockParameters x)),
      (BlockWord.bPointer (blockParameters x) s k).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true →
      (BlockWord.bPointer (blockParameters x) s k).address (TileShape.indexToList _ i)<x.b.capacity) ∧
    (∀ (i : BlockWord.CIndex (blockParameters x)), BlockWord.active (blockParameters x) s i →
      BlockWord.outputOffset (blockParameters x) s i<x.c.capacity) :=
  ⟨BlockWord.input_access _ s _ (input_bounds x h),
    BlockWord.weight_access _ s _ (weight_bounds x h),BlockWord.output_access _ s _ (output_bounds x h)⟩

theorem scaled_accesses (x : Arguments) (scale : Tensor) (h : Valid x) (hs : ScaleValid x scale)
    (hne : ¬emptyOutput x) (s : State Word) :
    (∀ k, k<(scaledParameters x scale).steps → ∀ (i : ScaledWord.AIndex (scaledParameters x scale)),
      ((scaledParameters x scale).even=Bool.true ∨
        ScaledWord.inputInner (scaledParameters x scale) s i.2.1<
        (scaledParameters x scale).inner-k*(scaledParameters x scale).blockInner) →
      ScaledWord.aOffset (scaledParameters x scale) s k i<x.a.capacity) ∧
    (∀ k, k<(scaledParameters x scale).steps → ∀ (i : ScaledWord.BIndex (scaledParameters x scale)),
      ((scaledParameters x scale).even=Bool.true ∨
        ScaledWord.inputInner (scaledParameters x scale) s i.1<
        (scaledParameters x scale).inner-k*(scaledParameters x scale).blockInner) →
      ScaledWord.bOffset (scaledParameters x scale) s k i<x.b.capacity) ∧
    (∀ (i : ScaledWord.CIndex (scaledParameters x scale)), ScaledWord.active (scaledParameters x scale) s i →
      ScaledWord.outputOffset (scaledParameters x scale) s i<x.c.capacity) ∧
    (∀ (i : ScaledWord.CIndex (scaledParameters x scale)), ScaledWord.active (scaledParameters x scale) s i →
      ScaledWord.outputRow (scaledParameters x scale) s i*(scaledParameters x scale).scaleRowStride<scale.capacity) := by
  have hM : 0<(scaledParameters x scale).rows := by
    change 0<x.a.rows; unfold emptyOutput at hne; omega
  have hN : 0<(scaledParameters x scale).columns := by
    change 0<x.b.columns; unfold emptyOutput at hne; omega
  exact ⟨ScaledWord.input_access _ s _ hM (input_bounds x h) (launch_even x scale h),
    ScaledWord.weight_access _ s _ hN (weight_bounds x h) (launch_even x scale h),
    ScaledWord.output_access _ s _ (output_bounds x h),ScaledWord.scale_access _ s _ (scale_bounds x scale hs)⟩

#axiomsClean input_bounds
#axiomsClean weight_bounds
#axiomsClean output_bounds
#axiomsClean scale_bounds
#axiomsClean block_accesses
#axiomsClean scaled_accesses
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Host

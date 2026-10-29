import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelPhysical

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelLowering
open VeriTile.Triton FP.Structural Kernel KernelWhole KernelTensor KernelPhysical
noncomputable section
open Classical
local instance : Inhabited Word := ⟨KernelModels.zero⟩
set_option maxHeartbeats 2500000

theorem output_no_scale (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (i : Nat) (hi : i<(m .output).numel) : ¬ScaleTouched m w (base m .output+i) := by
  intro hs
  have hb := scale_span m w h _ hs
  have sep := h.separate .output .scale (by decide)
  simp only [elementBytes] at sep
  omega

theorem scale_no_code (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (i : Nat) (b : Fin 4) (hi : i<(m .scale).numel) : ¬CodeTouched m w (base m .scale+4*i+b.val) := by
  intro hs
  have hb := code_span m w h _ hs
  have sep := h.separate .output .scale (by decide)
  have hb' := b.isLt
  simp only [elementBytes] at sep
  omega

/-- Every byte in the whole caller output span, including unwritten rows,
reifies the same source-final typed memory. Thus post is an actual store
lowering of the source result, not an independent expected-result heap. -/
theorem output_reifies (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (before : Heap) (final : State Word) (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → Word)
    (result : KernelWhole.Result m w codes scales (KernelTensor.initial m before) final)
    (i : Nat) (hi : i<(m .output).numel) :
    post m w before final (base m .output+i)=byteOfInt ((final.mem "O" i).read .int) := by
  by_cases ht : CodeTouched m w (base m .output+i)
  · simp [post,ht]
  have hs := output_no_scale m w h i hi
  have frame := result.2.2 "O" i (Or.inr (by
    intro pid g c hp hg hc he
    exact ht ⟨pid,g,c,hp,hg,hc,congrArg (fun x => base m .output+x) he⟩)) (Or.inl (by decide))
  rw [post_frame m w before final _ (by exact fun h => h.elim ht hs),frame]
  change before (base m .output+i)=byteOfInt (int8At before (base m .output+i)).toInt
  rw [byte_roundtrip]
  apply Fin.ext
  simp [byteOfBits,int8At]

/-- All four bytes of every scale cell reify the same source-final FP32 word,
including initial values on rows untouched by the whole grid. -/
theorem scale_reifies (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (before : Heap) (final : State Word) (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → Word)
    (result : KernelWhole.Result m w codes scales (KernelTensor.initial m before) final)
    (i : Nat) (b : Fin 4) (hi : i<(m .scale).numel) :
    post m w before final (base m .scale+4*i+b.val)=scaleByte ((final.mem "S" i).read .real) b := by
  have hc := scale_no_code m w h i b hi
  by_cases hs : ScaleTouched m w (base m .scale+4*i+b.val)
  · have hb := b.isLt
    have hd : (base m .scale+4*i+b.val-base m .scale)/4=i := by omega
    have hm : (base m .scale+4*i+b.val-base m .scale)%4=b.val := by omega
    simp only [post,if_neg hc,if_pos hs,hd]
    congr 1
    exact Fin.ext hm
  have frame := result.2.2 "S" i (Or.inl (by decide)) (Or.inr (by
    intro pid g hp hg he
    exact hs ⟨pid,g,b,hp,hg,congrArg (fun x => base m .scale+4*x+b.val) he⟩))
  rw [post_frame m w before final _ (by exact fun h => h.elim hc hs),frame]
  change before (base m .scale+4*i+b.val)=scaleByte (fp32At before (base m .scale+4*i)) b
  exact (encode_decode (fun b => before (base m .scale+4*i+b.val)) b).symm

/-- Complete physical frame also covers arbitrary addresses of shared backing
allocations, including gaps and elements outside all four Tensor data spans. -/
theorem backing_frame (t : Tensors) (w : Nat → MetadataWord) (before : Heap) (final : State Word)
    (r : Role) (offset : Nat) (miss : ¬KernelPhysical.Touched (metadata t) w ((t r).storage.base+offset)) :
    post (metadata t) w before final ((t r).storage.base+offset)=before ((t r).storage.base+offset) :=
  post_frame _ _ _ _ _ miss

#axiomsClean output_reifies
#axiomsClean scale_reifies
#axiomsClean backing_frame
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelLowering

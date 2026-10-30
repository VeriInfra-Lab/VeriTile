import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Native

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelAddresses
open VeriTile.Triton FP.Structural Kernel KernelWhole KernelTensor
noncomputable section
local instance : Inhabited Word := ⟨KernelModels.zero⟩
set_option maxHeartbeats 2500000
set_option maxRecDepth 8000

/-- Active typed offsets are exactly the source's signed INT64 index words. -/
theorem indices (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (pid g c : Nat) (hp : pid<(request m w).programs) (hg : g<(request m w).groups) (hc : c<8) :
    (Native.inputIndex (request m w) pid g c).toInt=(inputOffset (parameters m w) pid g c : Int) ∧
    (Native.outputIndex (request m w) ((request m w).destination pid) g c).toInt=
      (codeOffset (parameters m w) ((request m w).destination pid) g c : Int) := by
  simpa [inputOffset,codeOffset,parameters,groupWidth,Nat.add_mul,Nat.mul_assoc] using Native.active_index m w h pid g c hp hg hc

 theorem pointers (t : Tensors) (laws : ∀ r,Laws (t r)) (before : Heap)
    (h : InputContract (metadata t) (words t before)) (pid g c : Nat)
    (hp : pid<(request (metadata t) (words t before)).programs)
    (hg : g<(request (metadata t) (words t before)).groups) (hc : c<8) :
    let m := metadata t
    let w := words t before
    (Native.pointer (base m .input) (Native.inputIndex (request m w) pid g c) 4).toNat=
      base m .input+4*inputOffset (parameters m w) pid g c ∧
    (Native.pointer (base m .output) (Native.outputIndex (request m w) ((request m w).destination pid) g c) 1).toNat=
      base m .output+codeOffset (parameters m w) ((request m w).destination pid) g c ∧
    ∀ b : Fin 4,
    (Native.pointer (base m .scale) (Native.scaleIndex (request m w) ((request m w).destination pid) g) 4).toNat+b.val=
      base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g+b.val := by
  let m := metadata t
  let w := words t before
  have cap := span_capacity t laws h.contiguous
  have bi := KernelTensor.input_bound m w h pid g c hp hg hc
  have bo := KernelPhysical.output_bound m w h pid g c hp hg hc
  have bs := KernelPhysical.scale_bound m w h pid g hp hg
  have ci := cap .input
  have co := cap .output
  have cs := cap .scale
  simp only [elementBytes] at ci co cs
  have hi : Native.inputIndex (request m w) pid g c=Native.signed (inputOffset (parameters m w) pid g c) := by
    rw [Native.index_word]
    congr 1
    simp [inputOffset,parameters,groupWidth,Nat.add_mul,Nat.mul_assoc]
  have ho : Native.outputIndex (request m w) ((request m w).destination pid) g c=
      Native.signed (codeOffset (parameters m w) ((request m w).destination pid) g c) := by
    rw [Native.output_word]
    congr 1
    simp [codeOffset,parameters,groupWidth,Nat.add_mul,Nat.mul_assoc]
  have hs : Native.scaleIndex (request m w) ((request m w).destination pid) g=
      Native.signed (scaleOffset (parameters m w) ((request m w).destination pid) g) := by
    simp [Native.scaleIndex,Native.signed,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,scaleOffset,parameters]
  dsimp only
  simp only [m,w] at *
  rw [hi,ho]
  refine ⟨?_,?_,?_⟩
  · rw [Native.pointer_value _ _ _ (by omega)]
    omega
  · rw [Native.pointer_value _ _ _ (by omega)]
    omega
  · intro b
    rw [hs,Native.pointer_value _ _ _ (by omega)]
    omega

#axiomsClean indices
#axiomsClean pointers
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelAddresses

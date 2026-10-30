import bench.tritonbench_g.attention_fwd_triton3.Proofs.Tensor
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Physical
import bench.tritonbench_g.lightning_attention.ByteCodec

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Bytes
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly (encode decode load32 decode_encode)
noncomputable section
abbrev Word := BitVec 32
abbrev Byte := BitVec 8
abbrev Memory := Nat → Byte

/-- Publication reads the completed native region memory. No attention
formula or independently synthesized numerical result enters this function. -/
def publish (c : Call) (A : FlatAlloc) (final : RegionName → Nat → Word)
    (before : Memory) : Memory := by
  classical
  exact fun addr => if hit : ∃ idx lane b,active c lane ∧ Physical.address c A idx lane b=addr then
    let idx := Classical.choose hit
    let lane := Classical.choose (Classical.choose_spec hit)
    let b := Classical.choose (Classical.choose_spec (Classical.choose_spec hit))
    let p := Ownership.address c idx lane
    encode (final p.1 p.2) b
  else before addr

theorem readback (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : TensorBinding.Request) (admission : TensorBinding.Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (final : RegionName → Nat → Word) (before : Memory)
    (idx : GridIndex c.grid) (lane : Lane c.BM c.D) (b : Fin 4) (live : active c lane) :
    publish c A final before (Physical.address c A idx lane b)=
      encode (final (Ownership.address c idx lane).1 (Ownership.address c idx lane).2) b := by
  have safety := Contract.safety admission.checked
  have hit : ∃ jdx j e,active c j ∧ Physical.address c A jdx j e=Physical.address c A idx lane b :=
    ⟨idx,lane,b,live,rfl⟩
  rw [publish,dif_pos hit]
  have witness := Classical.choose_spec (Classical.choose_spec (Classical.choose_spec hit))
  obtain ⟨hidx,hlane,hbyte⟩ := Physical.unique c safety.valid A safety.allocated
    safety.disjoint closed _ idx _ lane _ b witness.2
  exact congrArg₂ encode
    (congrArg (fun addr => final addr.1 addr.2) (congrArg₂ (Ownership.address c) hidx hlane)) hbyte

theorem frame (c : Call) (A : FlatAlloc) (final : RegionName → Nat → Word)
    (before : Memory) (addr : Nat)
    (outside : ¬∃ idx lane b,active c lane ∧ Physical.address c A idx lane b=addr) :
    publish c A final before addr=before addr := by
  rw [publish,dif_neg outside]

theorem word_readback (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : TensorBinding.Request) (admission : TensorBinding.Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (final : RegionName → Nat → Word) (before : Memory)
    (idx : GridIndex c.grid) (lane : Lane c.BM c.D) (live : active c lane) :
    load32 (publish c A final before) (4*(Flat.address c A idx lane).2)=
      final (Ownership.address c idx lane).1 (Ownership.address c idx lane).2 := by
  have bytes : (fun b : Fin 4 => publish c A final before
      (4*(Flat.address c A idx lane).2+b.val))=
      encode (final (Ownership.address c idx lane).1 (Ownership.address c idx lane).2) := by
    funext b
    exact readback c A p r admission closed final before idx lane b live
  rw [load32,bytes,decode_encode]

/-- A logical element of the actual supplied output Tensor reads exactly the
word from final source memory, including all NaN payload and signed-zero bits.
The `coordinate` premise identifies a store location, never its value. -/
theorem tensor_readback (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : TensorBinding.Request) (admission : TensorBinding.Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (final : RegionName → Nat → Word) (before : Memory)
    (role : TensorBinding.Role) (i : Nat) (bound : i<(r role).tensor.metadata.numel)
    (idx : GridIndex c.grid) (lane : Lane c.BM c.D) (live : active c lane)
    (coordinate : Ownership.address c idx lane=
      (TensorBinding.region role,TensorBinding.offset (r role).tensor.metadata.shape (r role).strides i)) :
    Torch.FP32.logicalValue (fun addr => ⟨load32 (publish c A final before) addr⟩)
      (r role).tensor (admission.laws role).tensor i=
      ⟨final (Ownership.address c idx lane).1 (Ownership.address c idx lane).2⟩ := by
  rw [TensorBinding.all_six_inputs c A p r admission]
  have word := word_readback c A p r admission closed final before idx lane live
  simpa only [Flat.address,coordinate] using congrArg (fun bits => (⟨bits⟩ : Torch.FP32.Word)) word
  exact bound

#specNonCircular publish avoiding [FP.Structural.exec,FP.Structural.run,Raw.output]
#axiomsClean readback
#axiomsClean frame
#axiomsClean word_readback
#axiomsClean tensor_readback
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Bytes

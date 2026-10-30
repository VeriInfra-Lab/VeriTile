import bench.repairs.tritonbench_g.log_softmax.Proofs.Observed
import bench.repairs.tritonbench_g.log_softmax.Proofs.TensorBinding
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native

/-! Actual canonical source graphs, observed with a fixed native reduction and
fusion selection. The graph retains casts and load boundaries; a fused FP32
sub/mul site is interpreted by fmaRN, never by assuming split operations agree.
Compiler selection and individual instruction semantics are model parameters,
not whole-result or successful-execution hypotheses. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.NativeGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section
abbrev Word := VeriTile.Torch.FP32.Word

def entry (memory : RegionName → Nat → Word) : State Native.Term where
  mem := fun r i => .mk .real (.leaf r i (memory r i).bits)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

@[simp] theorem observe_entry (I : Native.Instructions) (S : Native.Schedule)
    (memory : RegionName → Nat → Word) (r : RegionName) (i : Nat) :
    Native.observe I S (entry memory) r i=(memory r i).bits := rfl

/-- Constructive execution of both full source grids, all-bit-pattern native
observation equality, and the complete exterior raw-word frame. -/
theorem whole (direction : Direction) (R : Native.Reductions)
    (I : Native.Instructions) (S : Native.Schedule) (M N K BM C : Nat) (hc : 0<C)
    (memory : RegionName → Nat → Word) :
    ∃ original commuted,
      Observed.run direction (Native.algebra R) Bool.false M N K BM C
        (KernelGrid.programs M N K BM) (entry memory)=some original ∧
      Observed.run direction (Native.algebra R) Bool.true M N K BM C
        (KernelGrid.programs M N K BM) (entry memory)=some commuted ∧
      (∀ (r : RegionName) i,Native.observe I S original r i=Native.observe I S commuted r i) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i →
        Native.observe I S original r i=(memory r i).bits ∧
        Native.observe I S commuted r i=(memory r i).bits) := by
  obtain ⟨final,original,commuted,frame⟩ := Observed.success_frame direction (Native.algebra R)
    M N K BM C hc (KernelGrid.programs M N K BM) (entry memory)
  refine ⟨final,final,original,commuted,fun _ _ => rfl,?_⟩
  intro r i outside
  have observed : Native.observe I S final r i=(memory r i).bits := by
    unfold Native.observe
    rw [frame r i outside]
    rfl
  exact ⟨observed,observed⟩

/-- The backward source's adjacent FP32 g - p*sum site can be selected as a
native fused operation. No equality with separately rounded arithmetic occurs. -/
theorem backward_fused (I : Native.Instructions) (S : Native.Schedule)
    (g p sum : Native.Term)
    (selected : S.fusion (.binary (some .fp32) .real .sub (Native.shape g)
      (.binary (some .fp32) .real .mul (Native.shape p) (Native.shape sum)))=.negativeLeftFirst) :
    Native.eval I S (.binary (some .fp32) .real .sub g (.binary (some .fp32) .real .mul p sum))=
      I.fmaRN (Native.eval I S p ^^^ 0x80000000) (Native.eval I S sum) (Native.eval I S g) := by
  simp [Native.eval,Native.shape,selected]

/-- Initial source words are bound to one absolute pointer memory. This allows
input aliases; it is a local read embedding, with no kernel-result premise. -/
def InputMemory (memory : RegionName → Nat → Word) (region : RegionName)
    (read : Nat → Word) (t : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws t) (hc : t.metadata.isContiguous=Bool.true)
    (ha : 0<t.metadata.numel → t.dataPtr.toNat%4=0) : Prop :=
  ∀ i,i<t.metadata.numel → memory region i=read ((Binding.dense t laws hc ha).address i)

/-- The graph leaves observed by the actual source carry the original caller's
logical FP32 values after the already-existing contiguous conversion. -/
theorem prepared_input (I : Native.Instructions) (S : Native.Schedule)
    (memory : RegionName → Nat → Word) (region : RegionName)
    (input : VeriTile.Torch.FP32.Tensor) (inputLaws : VeriTile.Torch.FP32.TensorLaws input)
    (before after : Nat → Word) (result : Binding.ContiguousResult input inputLaws before after)
    (binding : InputMemory memory region after result.tensor result.laws result.contiguous result.aligned)
    (i : Nat) (hi : i < input.metadata.numel) :
    Native.observe I S (entry memory) region i=
      (VeriTile.Torch.FP32.logicalValue before input inputLaws i).bits := by
  rw [observe_entry,binding i (by simpa only [VeriTile.Torch.TensorMetadata.numel,result.shape] using hi),
    Binding.prepared_read input inputLaws before after result i hi]

/-- Saved output is already contiguous in the backward Python contract. -/
theorem direct_input (I : Native.Instructions) (S : Native.Schedule)
    (memory : RegionName → Nat → Word) (region : RegionName) (read : Nat → Word)
    (t : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true) (hn : t.metadata.isNeg=Bool.false)
    (ha : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (binding : InputMemory memory region read t laws hc ha) (i : Nat) (hi : i<t.metadata.numel) :
    Native.observe I S (entry memory) region i=(VeriTile.Torch.FP32.logicalValue read t laws i).bits := by
  rw [observe_entry,binding i hi,Binding.direct_read t laws hc hn ha read i hi]

#axiomsClean whole
#axiomsClean backward_fused
#axiomsClean prepared_input
#axiomsClean direct_input
#specNonCircular Native.eval avoiding [Observed.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogSoftmax.NativeGrid

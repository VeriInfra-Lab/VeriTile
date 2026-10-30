import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelTensorBindings

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- All bytes in this output's complete logical write span. -/
def Written (v : OutputView c) (addr : Nat) : Prop :=
  v.tensor.dataPtr.toNat≤addr ∧ (addr-v.tensor.dataPtr.toNat)/4<c.outputSize

theorem written_inside (v : OutputView c) (addr : Nat) (h : Written v addr) :
    v.tensor.storage.base≤addr ∧ addr<v.tensor.storage.base+4*v.tensor.storage.elements := by
  let off := (addr-v.tensor.dataPtr.toNat)/4
  have bound : off<c.outputSize := h.2
  have coord := output_coordinates v off bound
  have physical := v.laws.inBounds off (by simpa only [output_numel] using bound)
  have remainder := Nat.mod_lt (addr-v.tensor.dataPtr.toNat) (by decide : 0<4)
  have split := Nat.div_add_mod (addr-v.tensor.dataPtr.toNat) 4
  unfold address Torch.FP32.Storage.address at coord
  have lower := h.1
  dsimp [off] at coord physical
  omega

def SharedWritten (f : Buffers c hasBias) (addr : Nat) : Prop := ∃ r,Written (f.output r) addr

theorem written_unique (f : Buffers c hasBias) (r s : Output) (addr : Nat)
    (hr : Written (f.output r) addr) (hs : Written (f.output s) addr) : r=s := by
  by_contra ne
  have sep := f.outputSeparate r s ne
  have br := written_inside (f.output r) addr hr
  have bs := written_inside (f.output s) addr hs
  unfold Apart at sep
  omega

/-- One backing for both outputs and every frame observation. Each changed
byte is encoded from the actual completed source state at its mapped address. -/
def post {α β : Type} [Inhabited α] (encode : α→Fin 4→β) (before : Nat→β)
    (f : Buffers c hasBias) (t : State α) (addr : Nat) : β := by
  classical
  exact if h : SharedWritten f addr then
    let r := Classical.choose h
    encode ((t.mem r.region ((addr-(f.output r).tensor.dataPtr.toNat)/4)).read .real)
      ⟨(addr-(f.output r).tensor.dataPtr.toNat)%4,Nat.mod_lt _ (by decide)⟩
  else before addr

theorem post_exterior {α β : Type} [Inhabited α] (encode : α→Fin 4→β) (before : Nat→β)
    (f : Buffers c hasBias) (t : State α) (addr : Nat) (outside : ¬SharedWritten f addr) :
    post encode before f t addr=before addr := by simp [post,outside]

theorem post_written {α β : Type} [Inhabited α] (encode : α→Fin 4→β) (before : Nat→β)
    (f : Buffers c hasBias) (t : State α) (r : Output) (addr : Nat)
    (written : Written (f.output r) addr) :
    post encode before f t addr=
      encode ((t.mem r.region ((addr-(f.output r).tensor.dataPtr.toNat)/4)).read .real)
        ⟨(addr-(f.output r).tensor.dataPtr.toNat)%4,Nat.mod_lt _ (by decide)⟩ := by
  classical
  have h : SharedWritten f addr := ⟨r,written⟩
  have he : Classical.choose h=r := written_unique f _ r addr (Classical.choose_spec h) written
  simp only [post,dif_pos h]
  simp only [he]

theorem post_coordinate {α β : Type} [Inhabited α] (encode : α→Fin 4→β) (before : Nat→β)
    (f : Buffers c hasBias) (t : State α) (r : Output) (off : Nat)
    (bound : off<c.outputSize) (byte : Fin 4) :
    post encode before f t (address (f.output r).tensor off+byte.val)=
      encode ((t.mem r.region off).read .real) byte := by
  have index : (address (f.output r).tensor off+byte.val-(f.output r).tensor.dataPtr.toNat)/4=off := by
    unfold address; omega
  have remainder : (address (f.output r).tensor off+byte.val-(f.output r).tensor.dataPtr.toNat)%4=byte.val := by
    unfold address; omega
  have written : Written (f.output r) (address (f.output r).tensor off+byte.val) := by
    refine ⟨?_,?_⟩
    · unfold address; omega
    · simpa only [index] using bound
  rw [post_written encode before f t r _ written,index]
  congr 1
  exact Fin.ext remainder

theorem post_input_frame {α β : Type} [Inhabited α] (encode : α→Fin 4→β) (before : Nat→β)
    (f : Buffers c hasBias) (t : State α) (input : Torch.FP32.Tensor)
    (member : input∈f.inputs.all) (addr : Nat)
    (inside : input.storage.base ≤ addr ∧ addr < input.storage.base+4*input.storage.elements) :
    post encode before f t addr=before addr := by
  apply post_exterior
  rintro ⟨r,hit⟩
  have bound := written_inside (f.output r) addr hit
  have sep := f.inputSeparate input member r
  unfold Apart at sep
  omega

/-- This scalar store writes precisely its four output bytes in the same
post-memory; intermediate/final values are taken from the shared interpreter. -/
theorem post_store {α β : Type} [Inhabited α] (encode : α→Fin 4→β) (before : Nat→β)
    (f : Buffers c hasBias) (t : State α) (r : Output) (off : Nat)
    (bound : off<c.outputSize) (value : α) (addr : Nat) :
    post encode before f (t.write r.region off (.mk .real value)) addr=
      if h : address (f.output r).tensor off≤addr ∧ addr<address (f.output r).tensor off+4 then
        encode value ⟨addr-address (f.output r).tensor off,by omega⟩
      else post encode before f t addr := by
  by_cases inside : address (f.output r).tensor off≤addr ∧ addr<address (f.output r).tensor off+4
  · let byte : Fin 4 := ⟨addr-address (f.output r).tensor off,by omega⟩
    have eq : addr=address (f.output r).tensor off+byte.val := by dsimp [byte]; omega
    rw [dif_pos inside]
    conv_lhs => rw [eq]
    rw [post_coordinate encode before f _ r off bound byte]
    simp [State.write,byte]
  · rw [dif_neg inside]
    by_cases hit : SharedWritten f addr
    · obtain ⟨s,hs⟩ := hit
      rw [post_written encode before f _ s addr hs,post_written encode before f t s addr hs]
      by_cases same : s=r
      · subst s
        have different : (addr-(f.output r).tensor.dataPtr.toNat)/4≠off := by
          have lower := hs.1
          unfold address at inside
          omega
        simp [State.write,different]
      · have different : s.region≠r.region := by
          cases s <;> cases r <;> simp_all [Output.region]
        simp [State.write,different]
    · rw [post_exterior encode before f _ addr hit,post_exterior encode before f t addr hit]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor

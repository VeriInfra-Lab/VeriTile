import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelTensorMemory
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelTensorAddress

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor
open VeriTile Triton FP.Structural
open Scalar
noncomputable section
set_option maxHeartbeats 4000000
abbrev Word := BitVec 32
abbrev Bytes := Nat→BitVec 8

/-- Actual little-endian FP32 loads from the common entry byte memory. -/
def read32 (heap : Bytes) (addr : Nat) : Word :=
  heap (addr+3) ++ heap (addr+2) ++ heap (addr+1) ++ heap addr
def wordByte (word : Word) (byte : Fin 4) : BitVec 8 := word.extractLsb' (8*byte.val) 8

theorem decode_encode (word : Word) :
    wordByte word 3 ++ wordByte word 2 ++ wordByte word 1 ++ wordByte word 0=word := by
  change (word.extractLsb' 24 8 ++ word.extractLsb' 16 8 ++
    word.extractLsb' 8 8 ++ word.extractLsb' 0 8)=word
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem read32_byte (heap : Bytes) (addr : Nat) (byte : Fin 4) :
    wordByte (read32 heap addr) byte=heap (addr+byte.val) := by
  fin_cases byte
  · exact BitVec.extractLsb'_append_eq_right
  · change ((heap (addr+3) ++ heap (addr+2) ++ heap (addr+1) ++ heap addr) : Word).extractLsb' 8 8=heap (addr+1)
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((heap (addr+3) ++ heap (addr+2) ++ heap (addr+1) ++ heap addr) : Word).extractLsb' 16 8=heap (addr+2)
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((heap (addr+3) ++ heap (addr+2) ++ heap (addr+1) ++ heap addr) : Word).extractLsb' 24 8=heap (addr+3)
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

def bytePost (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias) (t : State (Term Word)) : Bytes :=
  post (fun x byte=>wordByte (interpret M x) byte) before f t

/-- The common backing is exactly the original byte memory before execution.
Together with post_store this binds each source store to four physical bytes. -/
theorem byte_entry (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias) :
    bytePost M before f (entryState 0 (read32 before) f)=before := by
  funext addr
  by_cases hit : SharedWritten f addr
  · obtain ⟨r,hr⟩ := hit
    let off := (addr-(f.output r).tensor.dataPtr.toNat)/4
    let byte : Fin 4 := ⟨(addr-(f.output r).tensor.dataPtr.toNat)%4,Nat.mod_lt _ (by decide)⟩
    have eq : addr=address (f.output r).tensor off+byte.val := by
      have lower := hr.1
      have split := Nat.div_add_mod (addr-(f.output r).tensor.dataPtr.toNat) 4
      dsimp [address,off,byte]
      omega
    have bound : off<c.outputSize := hr.2
    rw [eq]
    change post _ before f _ _=before _
    rw [post_coordinate _ before f _ r off bound byte]
    cases r
    · simpa [entryState,Output.region,Cell.read,interpret] using
        read32_byte before (address (f.output .prepared).tensor off) byte
    · simpa [entryState,Output.region,Cell.read,interpret] using
        read32_byte before (address (f.output .cumulative).tensor off) byte
  · exact post_exterior _ before f _ addr hit

theorem tensor_byte (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias)
    (t : State (Term Word)) (r : Output) (off : Nat) (bound : off<c.outputSize) (byte : Fin 4) :
    bytePost M before f t ((f.output r).tensor.storage.address ((f.output r).tensor.physicalIndex off)+byte.val)=
      wordByte (interpret M ((t.mem r.region off).read .real)) byte := by
  rw [output_coordinates (f.output r) off bound]
  exact post_coordinate _ before f t r off bound byte

/-- Decoding those same four bytes gives the Tensor word, including every
NaN payload, signed zero, subnormal and infinity bit. -/
theorem tensor_word (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias)
    (t : State (Term Word)) (r : Output) (off : Nat) (bound : off<c.outputSize) :
    logical (read32 (bytePost M before f t)) (f.output r).tensor off=
      interpret M ((t.mem r.region off).read .real) := by
  have bytes := tensor_byte M before f t r off bound
  unfold logical read32
  have b0 := bytes 0
  simp only [Fin.val_zero,Nat.add_zero] at b0
  rw [bytes ⟨3,by decide⟩,bytes ⟨2,by decide⟩,bytes ⟨1,by decide⟩,b0]
  exact decode_encode _

theorem byte_input_frame (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias)
    (t : State (Term Word)) (input : Torch.FP32.Tensor) (member : input∈f.inputs.all)
    (addr : Nat) (inside : input.storage.base ≤ addr ∧ addr < input.storage.base+4*input.storage.elements) :
    bytePost M before f t addr=before addr := post_input_frame _ before f t input member addr inside

theorem input_word_frame (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias)
    (t : State (Term Word)) (input : Torch.FP32.Tensor) (member : input∈f.inputs.all)
    (laws : Torch.FP32.TensorLaws input) (i : Nat) (bound : i < input.metadata.numel) :
    logical (read32 (bytePost M before f t)) input i=logical (read32 before) input i := by
  have physical := laws.inBounds i bound
  have bytes (byte : Fin 4) :
      bytePost M before f t (input.storage.address (input.physicalIndex i)+byte.val)=
        before (input.storage.address (input.physicalIndex i)+byte.val) := by
    apply byte_input_frame M before f t input member
    unfold Torch.FP32.Storage.address
    omega
  have b0 := bytes 0
  simp only [Fin.val_zero,Nat.add_zero] at b0
  unfold logical read32
  rw [bytes ⟨3,by decide⟩,bytes ⟨2,by decide⟩,bytes ⟨1,by decide⟩,b0]

def ByteFrame (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias) (t : State (Term Word)) : Prop :=
  (∀ addr,¬SharedWritten f addr → bytePost M before f t addr=before addr) ∧
  (∀ input∈f.inputs.all, ∀ addr,
    input.storage.base ≤ addr ∧ addr < input.storage.base+4*input.storage.elements →
      bytePost M before f t addr=before addr)

theorem byte_frame (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias)
    (t : State (Term Word)) : ByteFrame M before f t :=
  ⟨post_exterior _ before f t,byte_input_frame M before f t⟩

/-- Both actual returned Tensor views and all their bytes, in one derived
backing. Inputs are read from that backing's initial bytes. -/
def NativeOutputs (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias)
    (plan : PrefixPlan c.BH c.BC) (useSoftplus : Bool) (lower upper : Word) (t : State (Term Word)) : Prop :=
  ∀ b h k j, b<c.batches → h<c.heads → k<c.chunks → ∀ hj : j<c.chunk,
    let off := packed c b h k j
    let a := snapshot 0 (read32 before) f.inputs
    let prepared := referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k j
    let cumulative := referenceLogicalCumulative M c plan a hasBias useSoftplus lower upper b h k
      ⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩
    (Torch.FP32.logicalValue (fun addr=>⟨read32 (bytePost M before f t) addr⟩)
      (f.output .prepared).tensor (f.output .prepared).laws off).bits=prepared ∧
    (Torch.FP32.logicalValue (fun addr=>⟨read32 (bytePost M before f t) addr⟩)
      (f.output .cumulative).tensor (f.output .cumulative).laws off).bits=cumulative ∧
    (∀ byte : Fin 4,bytePost M before f t
      ((f.output .prepared).tensor.storage.address ((f.output .prepared).tensor.physicalIndex off)+byte.val)=wordByte prepared byte) ∧
    (∀ byte : Fin 4,bytePost M before f t
      ((f.output .cumulative).tensor.storage.address ((f.output .cumulative).tensor.physicalIndex off)+byte.val)=wordByte cumulative byte)

theorem native_outputs (M : NativeModel Word) (before : Bytes) (f : Buffers c hasBias)
    (plan : PrefixPlan c.BH c.BC) (useSoftplus : Bool) (lower upper : Word) (t : State (Term Word))
    (values : LogicalOutputs M c plan (snapshot 0 (read32 before) f.inputs) hasBias useSoftplus lower upper t) :
    NativeOutputs M before f plan useSoftplus lower upper t := by
  intro b h k j hb hh hk hj
  have bound := packed_bound c b h k j hb hh hk hj
  have value := values b h k j hb hh hk hj
  dsimp only
  refine ⟨?_,?_,?_,?_⟩
  · rw [native_logical _ _ _ (f.output .prepared).resolved,tensor_word M before f t .prepared _ bound]
    exact value.1
  · rw [native_logical _ _ _ (f.output .cumulative).resolved,tensor_word M before f t .cumulative _ bound]
    exact value.2
  · intro byte
    rw [tensor_byte M before f t .prepared _ bound byte]
    exact congrArg (fun w=>wordByte w byte) value.1
  · intro byte
    rw [tensor_byte M before f t .cumulative _ bound byte]
    exact congrArg (fun w=>wordByte w byte) value.2

theorem native_complete (M : NativeModel Word) (c : Geometry) (sizes : Sizes c)
    (before : Bytes) (f : Buffers c hasBias) (plan : PrefixPlan c.BH c.BC)
    (useSoftplus : Bool) (lower upper : Word) :
    ∃ t,Kernel.launch (graphAlgebra M c plan) c hasBias useSoftplus (.input lower) (.input upper)
        (entryState 0 (read32 before) f)=some t ∧
      NativeOutputs M before f plan useSoftplus lower upper t ∧ ByteFrame M before f t ∧
      Kernel.Frame c c.grid (entryState 0 (read32 before) f) t := by
  obtain ⟨t,ran,values,frame⟩ := Scalar.complete_from M c sizes plan
    (snapshot 0 (read32 before) f.inputs) (entryState 0 (read32 before) f)
    (entry_graph 0 (read32 before) f) hasBias useSoftplus lower upper
  exact ⟨t,ran,native_outputs M before f plan useSoftplus lower upper t values,
    byte_frame M before f t,frame⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor

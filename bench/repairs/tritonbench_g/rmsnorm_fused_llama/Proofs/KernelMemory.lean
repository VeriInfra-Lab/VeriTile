import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeSource
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Bytes

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelMemory
open VeriTile Triton FP.Structural Contract KernelAPI
open VeriTile.Bench.Repairs.SwigluTriton.Memory
namespace Raw
abbrev Memory := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.Memory
abbrev ops := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.ops
abbrev read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.read
abbrev unpack := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.unpack
abbrev byte_of_read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.byte_of_read
end Raw
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
variable {x w out : TensorInfo} {eps : Epsilon}

/-- Actual contiguous output pointer plus its element offset. -/
def key (t : Torch.FP32.Tensor) (i : Nat) : Key :=
  ⟨t.metadata.device.index,t.storage.address (t.storageOffset+i)⟩

def wordRead (heap : Raw.Memory) : Heap Native.Word := fun owner address=>Raw.read heap ⟨owner,address⟩

def logicalWord (heap : Raw.Memory) (t : Torch.FP32.Tensor) (laws : Torch.FP32.TensorLaws t) (i : Nat) : Torch.FP32.Word :=
  Torch.FP32.logicalValue (fun a=>⟨wordRead heap t.metadata.device.index a⟩) t laws i

theorem logical_word (heap : Raw.Memory) (t : Torch.FP32.Tensor) (laws : Torch.FP32.TensorLaws t)
    (noNeg : t.metadata.isNeg=Bool.false) (i : Nat) :
    (logicalWord heap t laws i).bits=tensorRead (wordRead heap) t i := by
  simp [logicalWord,Torch.FP32.logicalValue,noNeg,tensorRead]

def outputBytes (q : Host.Request) (b : Binding x w out) (words : Nat → Native.Word) (before : Raw.Memory) : Raw.Memory :=
  stores Raw.ops (key b.output) words (List.range q.size) before

/-- Memory update specified independently by each logical result. -/
def referenceBytes (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (q : Host.Request) (b : Binding x w out) (epsilon : Native.Word) (before : Raw.Memory) : Raw.Memory :=
  outputBytes q b (fun i=>RuntimeSource.nativeTarget I R S (wordRead before) b q epsilon (i/q.columns) (i%q.columns)) before

theorem stores_congr {M α β ι : Type} (ops : Ops M α β) (key : ι → Key) (v w : ι → α)
    (jobs : List ι) (same : ∀ i∈jobs,v i=w i) (before : M) :
    stores ops key v jobs before=stores ops key w jobs before := by
  induction jobs generalizing before with
  | nil=>rfl
  | cons i rest ih=>
    simp only [stores,same i List.mem_cons_self]
    exact ih (fun j hj=>same j (List.mem_cons_of_mem _ hj)) _

theorem key_injective (t : Torch.FP32.Tensor) : Function.Injective (key t) := by
  intro i j same
  have h := congrArg Key.address same
  simp only [key,Torch.FP32.Storage.address] at h
  omega

theorem key_disjoint (t : Torch.FP32.Tensor) (i j : Nat) (different : i≠j) : (key t i).disjoint (key t j) := by
  simp only [key,Key.disjoint,Torch.FP32.Storage.address]
  by_cases h : i<j
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

theorem bytes_read (q : Host.Request) (b : Binding x w out) (words : Nat → Native.Word)
    (before : Raw.Memory) (i : Nat) (hi : i<q.size) :
    Raw.read (outputBytes q b words before) (key b.output i)=words i := by
  apply readback Raw.ops (key b.output) words (List.range q.size) before i (List.mem_range.mpr hi)
  · intro j _ same
    rw [key_injective b.output same]
  · intro j _ different
    exact key_disjoint b.output i j (fun same=>different (congrArg (key b.output) same.symm))

theorem bytes_value (q : Host.Request) (b : Binding x w out) (words : Nat → Native.Word)
    (before : Raw.Memory) (i : Nat) (hi : i<q.size) (byte : Fin 4) :
    outputBytes q b words before (key b.output i).owner ((key b.output i).address+byte.val)=Raw.unpack (words i) byte :=
  (Raw.byte_of_read _ _ byte).trans (congrArg (fun word=>Raw.unpack word byte) (bytes_read q b words before i hi))

def Outside (q : Host.Request) (b : Binding x w out) (owner address : Nat) : Prop :=
  ∀ i,i<q.size → ¬(key b.output i).covers owner address

theorem bytes_frame (q : Host.Request) (b : Binding x w out) (words : Nat → Native.Word)
    (before : Raw.Memory) (owner address : Nat) (outside : Outside q b owner address) :
    outputBytes q b words before owner address=before owner address :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.bytes_frame Raw.ops _ _ _ _ _ _
    (fun i hi=>outside i (List.mem_range.mp hi))

theorem native_memory (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (q : Host.Request) (b : Binding x w out) (epsilon : Native.Word) (before : Raw.Memory) (final : State Native.Term)
    (values : ∀ row col,row<q.rows → col<q.columns →
      Native.observe I S final "y" (row*q.columns+col)=RuntimeSource.nativeTarget I R S (wordRead before) b q epsilon row col) :
    outputBytes q b (Native.observe I S final "y") before=referenceBytes I R S q b epsilon before := by
  apply stores_congr
  intro i hi
  have hi' := List.mem_range.mp hi
  have hr : i/q.columns<q.rows := (Nat.div_lt_iff_lt_mul q.positive).2 hi'
  have hc := Nat.mod_lt i q.positive
  have eq : i/q.columns*q.columns+i%q.columns=i := by
    simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div i q.columns
  exact (congrArg (Native.observe I S final "y") eq.symm).trans (values _ _ hr hc)

def writeCell (before : Heap ℝ) (k : Key) (value : ℝ) : Heap ℝ := fun owner address=>
  if owner=k.owner ∧ address=k.address then value else before owner address

def realOps : Ops (Heap ℝ) ℝ ℝ where
  read := fun s k=>s k.owner k.address
  write := writeCell
  observe := id
  read_same := by intro s k v;simp [writeCell]
  read_other := by
    intro s writer v observed apart
    unfold writeCell
    rw [if_neg]
    rintro ⟨ho,ha⟩
    apply Key.disjoint_ne apart
    cases observed;cases writer;simp_all
  frame := by
    intro s writer v owner address outside
    change writeCell s writer v owner address=s owner address
    unfold writeCell
    rw [if_neg]
    rintro ⟨ho,ha⟩
    apply outside
    simp [Key.covers,ho,ha]

def outputReals (q : Host.Request) (b : Binding x w out) (values : Nat → ℝ) (before : Heap ℝ) : Heap ℝ :=
  stores realOps (key b.output) values (List.range q.size) before

def referenceReals (q : Host.Request) (b : Binding x w out) (before : Heap ℝ) : Heap ℝ :=
  outputReals q b (fun i=>mathematical (tensorRead before b.input) (tensorRead before b.weight)
    (i/q.columns) q.columns q.epsilon (i%q.columns)) before

theorem real_read (q : Host.Request) (b : Binding x w out) (values : Nat → ℝ)
    (before : Heap ℝ) (i : Nat) (hi : i<q.size) :
    outputReals q b values before (key b.output i).owner (key b.output i).address=values i := by
  apply readback realOps (key b.output) values (List.range q.size) before i (List.mem_range.mpr hi)
  · intro j _ same;rw [key_injective b.output same]
  · intro j _ different
    exact key_disjoint b.output i j (fun same=>different (congrArg (key b.output) same.symm))

theorem real_frame (q : Host.Request) (b : Binding x w out) (values : Nat → ℝ)
    (before : Heap ℝ) (owner address : Nat) (outside : Outside q b owner address) :
    outputReals q b values before owner address=before owner address :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.bytes_frame realOps _ _ _ _ _ _
    (fun i hi=>outside i (List.mem_range.mp hi))

theorem real_memory (q : Host.Request) (b : Binding x w out) (before : Heap ℝ) (final : BlockState)
    (values : ∀ row col,row<q.rows → col<q.columns →
      final.readMem "y" (row*q.columns+col)=
        mathematical (tensorRead before b.input) (tensorRead before b.weight) row q.columns q.epsilon col) :
    outputReals q b (final.readMem "y") before=referenceReals q b before := by
  apply stores_congr
  intro i hi
  have hi' := List.mem_range.mp hi
  have hr : i/q.columns<q.rows := (Nat.div_lt_iff_lt_mul q.positive).2 hi'
  have hc := Nat.mod_lt i q.positive
  have eq : i/q.columns*q.columns+i%q.columns=i := by
    simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div i q.columns
  exact (congrArg (final.readMem "y") eq.symm).trans (values _ _ hr hc)

#axiomsClean bytes_read
#axiomsClean native_memory
#axiomsClean real_memory
#axiomsClean logical_word
#guard_msgs (drop info) in
#specNonCircular referenceBytes avoiding [kernel,runtimeKernel,KernelSource.entry,RuntimeGrid.run,RuntimeGrid.graph,RuntimeRow.value]
#guard_msgs (drop info) in
#specNonCircular referenceReals avoiding [kernel,RealGrid.run,RealGrid.value,KernelSource.realEntry]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelMemory

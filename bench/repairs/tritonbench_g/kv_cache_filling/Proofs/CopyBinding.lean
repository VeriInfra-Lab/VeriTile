import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.CopyExecution
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Tensor

namespace VeriTile.Bench.Repairs.KvCacheFilling.CopyBinding
open VeriTile.Triton FP.Structural
noncomputable section

def parameters (r : Request) : CopySource.Parameters :=
  ⟨r.dk, r.dv, r.ck, r.cv, r.ck.nextPowerOfTwo, r.cv.nextPowerOfTwo⟩

def inputTensor (t : TensorInputs) (key : Bool) : TensorDescriptor :=
  if key then t.key else t.value

/-- The current admitted Tensor observations fix the physical source address.
This representation law has no kernel, result, output, or numerical premise. -/
structure InputReadLaw {α β : Type} (t : TensorInputs) (heap : Nat → β)
    (read : β → α) (logical : Bool → Nat → α) : Prop where
  reads : ∀ key i, i < (inputTensor t key).metadata.numel →
    read (heap ((inputTensor t key).metadata.dataPtr.toNat + 4 * i)) = logical key i

/-- Kernel entry state from the actual payload Tensor pointers and the actual
INT64 job-array pointers. The same payload heap supplies every payload role.
Neither inputs nor output cells are synthesized from the mathematical target. -/
def tensorState {α β : Type} (t : TensorInputs) (heap : Nat → β) (read : β → α)
    (schedule : Nat → BitVec 64) (srcPtr dstPtr : Nat) (scratch : State α) : State α :=
  { scratch with mem := fun reg i =>
      if reg = "src" then .mk .int (schedule (srcPtr + 8 * i)).toInt
      else if reg = "dst" then .mk .int (schedule (dstPtr + 8 * i)).toInt
      else if reg = "key" then .mk .real (read (heap (t.key.metadata.dataPtr.toNat + 4 * i)))
      else if reg = "value" then .mk .real (read (heap (t.value.metadata.dataPtr.toNat + 4 * i)))
      else if reg = "keyCache" then .mk .real (read (heap (t.keyCache.metadata.dataPtr.toNat + 4 * i)))
      else if reg = "valueCache" then .mk .real (read (heap (t.valueCache.metadata.dataPtr.toNat + 4 * i)))
      else scratch.mem reg i }

theorem shape (r : Request) (d : Domain r) (hp : r.policy = .copy) :
    ∀ key, CopyExecution.capacity (parameters r) key = CopyExecution.width (parameters r) key := by
  have hk := d.key_shape
  have hv := d.value_shape
  simp [Request.factor, hp] at hk hv
  intro key
  cases key <;> simp [CopyExecution.capacity, CopyExecution.width, parameters, hk, hv]

theorem covered (r : Request) (d : Domain r) (hp : r.policy = .copy) :
    ∀ key, CopyExecution.width (parameters r) key ≤ CopyExecution.lanes (parameters r) key := by
  intro key
  rw [← shape r d hp key]
  cases key <;> exact VeriTile.Bench.Repairs.EmbeddingTritonKernel.Bounds.nextPower_covers _

theorem input_bound (t : TensorInputs) (r : Request) (h : InputContract t r)
    (j : Job) (hj : j ∈ jobs r) (key : Bool) (i : Nat)
    (hi : i < CopyExecution.width (parameters r) key) :
    sourceRow r j * CopyExecution.width (parameters r) key + i < (inputTensor t key).metadata.numel := by
  have hs := sourceRow_bound r (Native.domain h) j hj
  have hb := flat_bound (sourceRow r j) i (r.tokens * r.heads)
    (CopyExecution.width (parameters r) key) hs hi
  cases key <;>
    simpa [inputTensor, CopyExecution.width, parameters, VeriTile.Torch.TensorMetadata.numel,
      h.1.1, h.1.2.1, Nat.mul_assoc] using hb

/-- Successful policy-0 source/candidate executions composed with the same
admitted tensors, physical input observations and schedule loads. Both logical
cache rows and the complete logical frame are conclusions. This is still a
branch result: policies 4/8 and the final public kernel endpoints remain open. -/
theorem tensor_copy {α β : Type} [Inhabited α] (M : Algebra α)
    (t : TensorInputs) (r : Request) (h : InputContract t r) (hp : r.policy = .copy)
    (heap : Nat → β) (read : β → α) (logical : Bool → Nat → α)
    (law : InputReadLaw t heap read logical)
    (schedule : Nat → BitVec 64) (srcPtr dstPtr : Nat) (scratch : State α)
    (j : Job) (hj : j ∈ jobs r)
    (hs : (schedule (srcPtr + 8 * scratch.pids 0)).toInt = (sourceRow r j : Int))
    (hd : (schedule (dstPtr + 8 * scratch.pids 0)).toInt = (destinationRow r j : Int)) :
    let p := parameters r
    let initial := tensorState t heap read schedule srcPtr dstPtr scratch
    ∃ final,
      exec M (CopySource.originalKernel p) initial = some final ∧
      exec M (CopySource.commutedKernel p) initial = some final ∧
      (∀ key i, i < CopyExecution.width p key →
        final.mem (CopyExecution.output key) (destinationRow r j * CopyExecution.width p key + i) =
          .mk .real (M.fp32Load (logical key (sourceRow r j * CopyExecution.width p key + i)))) ∧
      (∀ reg i,
        (∀ key, reg ≠ CopyExecution.output key ∨ i < destinationRow r j * CopyExecution.width p key ∨
          destinationRow r j * CopyExecution.width p key + CopyExecution.width p key ≤ i) →
        final.mem reg i = initial.mem reg i) := by
  dsimp only
  let initial := tensorState t heap read schedule srcPtr dstPtr scratch
  have hs' : (initial.mem "src" (initial.pids 0)).read .int = (sourceRow r j : Int) := by
    simpa [initial, tensorState] using hs
  have hd' : (initial.mem "dst" (initial.pids 0)).read .int = (destinationRow r j : Int) := by
    simpa [initial, tensorState] using hd
  refine ⟨CopyExecution.result M (parameters r) initial (sourceRow r j) (destinationRow r j),
    CopyExecution.execute M _ _ _ _ hs' hd' Bool.false,
    CopyExecution.execute M _ _ _ _ hs' hd' Bool.true, ?_, ?_⟩
  · intro key i hi
    rw [CopyExecution.result_read M _ _ _ _ (shape r (Native.domain h) hp)
      (covered r (Native.domain h) hp) key i hi]
    have bound := law.reads key _ (input_bound t r h j hj key i hi)
    cases key <;> simpa [initial, tensorState, CopyExecution.input, inputTensor] using
      congrArg (fun x => FP.Structural.Cell.mk .real (M.fp32Load x)) bound
  · intro reg i outside
    exact CopyExecution.result_frame M _ _ _ _ (shape r (Native.domain h) hp) reg i outside

#axiomsClean tensor_copy
#specNonCircular InputReadLaw avoiding [tensorState, CopySource.originalKernel, CopyExecution.result]
end
end VeriTile.Bench.Repairs.KvCacheFilling.CopyBinding

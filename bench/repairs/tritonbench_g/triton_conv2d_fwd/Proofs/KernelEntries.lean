import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelExecution
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReal
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelStorage
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelTensor
namespace VeriTile.Bench.Repairs.Conv2d.KernelEntries
open VeriTile.Triton FP.Structural KernelTensor KernelBinding KernelNumeric
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16384
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

def physicalMemory (p : Prepared) (heap : Nat → α) : Memory α
  | .input,i => heap (p.call.input.pointer+4*i)
  | .weight,i => heap (p.call.weight.pointer+4*i)
  | .output,i => heap (p.output.pointer+4*i)
  | .other _,_ => default

def initial (p : Prepared) (heap : Nat → α) : State (KernelNumeric.Value α) where
  mem := fun r i=>.mk .fp32 (.scalar (if r="input" then heap (p.call.input.pointer+4*i)
    else if r="weight" then heap (p.call.weight.pointer+4*i)
    else if r="output" then heap (p.output.pointer+4*i) else default))
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun k=>if k=0 then blocks p.call.shape.spatial 128
    else if k=1 then blocks p.call.shape.outputs 32 else if k=2 then p.call.shape.groups else 0
  undef := fun d _ _=>defaultValue d

theorem initial_memory (o : Operations α) (p : Prepared) (heap : Nat → α) :
    KernelNumeric.memory o (initial p heap)=physicalMemory p heap := by
  funext r a
  cases r <;> simp [KernelNumeric.memory,KernelOperands.memory,initial,physicalMemory,scalar]

def realTarget (p : Prepared) (heap : Nat → ℝ) (x : Conv2d.Cell) :=
  mathematical p.call.shape (physicalMemory p heap) x

def nativeTarget (instructions : KernelNative.Instructions) (p : Prepared)
    (heap : KernelStorage.Heap) (x : Conv2d.Cell) : Native.Word :=
  KernelNumeric.reference (KernelNative.operations instructions p.call.flags) p.call.shape p.call.flags
    (physicalMemory p (KernelStorage.read heap)) x

def nativeHeap (instructions : KernelNative.Instructions) (p : Prepared)
    (before : KernelStorage.Heap) (final : State KernelStorage.Value) :=
  KernelStorage.image (KernelNative.operations instructions p.call.flags)
    p.output.pointer p.call.shape.size before final.mem

/-- The ideal carrier observes values at element addresses; the native carrier
below observes every byte of the same output Tensor's actual storage. -/
def realHeap (p : Prepared) (before : Nat → ℝ) (final : State (KernelNumeric.Value ℝ)) : Nat → ℝ := fun q=>
  if p.output.pointer≤q ∧ q<p.output.pointer+4*p.call.shape.size then
    scalar KernelReal.operations ((final.mem "output" ((q-p.output.pointer)/4)).read .fp32)
  else before q

def realResult (p : Prepared) (before : Nat → ℝ) (final : State (KernelNumeric.Value ℝ)) (x : Conv2d.Cell) :=
  KernelTensor.realPoint (realHeap p before final) p.result
    (batch p.call.shape x.spatial) (x.group*p.call.shape.og+x.feature)
    (outH p.call.shape x.spatial) (outW p.call.shape x.spatial)

def nativeResult (instructions : KernelNative.Instructions) (p : Prepared)
    (before : KernelStorage.Heap) (final : State KernelStorage.Value) (x : Conv2d.Cell) :=
  KernelTensor.nativePoint (fun a=>Software.value (KernelStorage.read (nativeHeap instructions p before final) a))
    p.result (batch p.call.shape x.spatial) (x.group*p.call.shape.og+x.feature)
      (outH p.call.shape x.spatial) (outW p.call.shape x.spatial)

theorem output_bounds (p : Prepared) (x : Conv2d.Cell) (hx : p.call.shape.Active x) :
    batch p.call.shape x.spatial<p.call.shape.n ∧
    x.group*p.call.shape.og+x.feature<p.call.shape.outputs ∧
    outH p.call.shape x.spatial<p.call.shape.oh ∧ outW p.call.shape x.spatial<p.call.shape.ow := by
  refine ⟨?_,feature_bound _ _ hx,Nat.mod_lt _ p.domain.valid.oh,Nat.mod_lt _ p.domain.valid.ow⟩
  rw [batch_eq]
  exact (Nat.div_lt_iff_lt_mul (plane_pos _ p.domain.valid)).mpr hx.2.1

theorem output_address (p : Prepared) (x : Conv2d.Cell) :
    (outputView p.call.shape).offset (batch p.call.shape x.spatial)
      (x.group*p.call.shape.og+x.feature) (outH p.call.shape x.spatial) (outW p.call.shape x.spatial)=
      outputPointer p.call.shape x := by
  unfold outputView View4.offset outputPointer
  ring

theorem native_initial (instructions : KernelNative.Instructions) (p : Prepared) (before : KernelStorage.Heap) :
    nativeHeap instructions p before (initial p (KernelStorage.read before))=before := by
  apply KernelStorage.image_initial
  intro i hi
  simp [initial]

theorem real_correct (p : Prepared) (heap : Nat → ℝ) :
    ∃ final,KernelGrid.launch (algebra KernelReal.operations) (halfDot KernelReal.operations)
      (defaultDot KernelReal.operations) p.call.shape p.call.flags (initial p heap)=some final ∧
      (∀ x,p.call.shape.Active x → realResult p heap final x=realTarget p heap x) ∧
      (∀ (r : RegionName) a,r≠"output" ∨ p.call.shape.size≤a → final.mem r a=(initial p heap).mem r a) := by
  obtain ⟨final,he,hi,hr,hf⟩:=launch_reference KernelReal.operations p.call.shape p.domain.valid p.call.flags (initial p heap)
  refine ⟨final,he,?_,hf⟩
  intro x hx
  have bounds:=output_bounds p x hx
  rw [realResult,←KernelTensor.real_read _ p.result _ _ _ _ bounds.1 bounds.2.1 bounds.2.2.1 bounds.2.2.2,
    output_address]
  have hb:=address_bound _ p.domain.valid x hx
  have inside : p.output.pointer≤p.output.pointer+4*outputPointer p.call.shape x ∧
      p.output.pointer+4*outputPointer p.call.shape x<p.output.pointer+4*p.call.shape.size := by omega
  have div : (p.output.pointer+4*outputPointer p.call.shape x-p.output.pointer)/4=outputPointer p.call.shape x := by omega
  simp only [realHeap,if_pos inside,div,hr x hx,FP.Structural.Cell.read_same,scalar]
  rw [initial_memory,KernelReal.reference_real _ _ _ _ hx]
  rfl

theorem native_correct (instructions : KernelNative.Instructions) (p : Prepared) (before : KernelStorage.Heap) :
    ∃ final,KernelGrid.launch (algebra (KernelNative.operations instructions p.call.flags))
      (halfDot (KernelNative.operations instructions p.call.flags))
      (defaultDot (KernelNative.operations instructions p.call.flags)) p.call.shape p.call.flags
      (initial p (KernelStorage.read before))=some final ∧
      (∀ x,p.call.shape.Active x → nativeResult instructions p before final x=Software.value (nativeTarget instructions p before x)) ∧
      (∀ x,p.call.shape.Active x → ∀ k : Fin 4,
        nativeHeap instructions p before final (p.output.pointer+4*outputPointer p.call.shape x+k.val)=nativeTarget instructions p before x k) ∧
      (∀ q,q<p.output.pointer ∨ p.output.pointer+4*p.call.shape.size≤q → nativeHeap instructions p before final q=before q) ∧
      (∀ q,q<p.call.input.capacity → nativeHeap instructions p before final (p.call.input.storageBase+q)=before (p.call.input.storageBase+q)) ∧
      (∀ q,q<p.call.weight.capacity → nativeHeap instructions p before final (p.call.weight.storageBase+q)=before (p.call.weight.storageBase+q)) := by
  let o:=KernelNative.operations instructions p.call.flags
  obtain ⟨final,he,hi,hr,hf⟩:=launch_reference o p.call.shape p.domain.valid p.call.flags (initial p (KernelStorage.read before))
  have frame (q : Nat) (hq : q<p.output.pointer ∨ p.output.pointer+4*p.call.shape.size≤q) :
      nativeHeap instructions p before final q=before q := KernelStorage.image_frame o _ _ _ _ q hq
  refine ⟨final,he,?_,?_,frame,?_,?_⟩
  · intro x hx
    have bounds:=output_bounds p x hx
    rw [nativeResult,←KernelTensor.native_read _ p.result p.allocation.kind _ _ _ _ bounds.1 bounds.2.1 bounds.2.2.1 bounds.2.2.2,
      output_address]
    apply congrArg Software.value
    funext k
    change KernelStorage.image o _ _ _ _ (p.output.pointer+4*outputPointer p.call.shape x+k.val)=_
    rw [KernelStorage.image_byte o _ _ _ _ _ (address_bound _ p.domain.valid x hx),hr x hx]
    simp only [FP.Structural.Cell.read_same,scalar]
    rw [initial_memory]
    rfl
  · intro x hx k
    change KernelStorage.image o _ _ _ _ (p.output.pointer+4*outputPointer p.call.shape x+k.val)=_
    rw [KernelStorage.image_byte o _ _ _ _ _ (address_bound _ p.domain.valid x hx),hr x hx]
    simp only [FP.Structural.Cell.read_same,scalar]
    rw [initial_memory]
    rfl
  · intro q hq
    apply frame
    have sep:=p.allocation.input
    unfold Physical.Separated at sep
    omega
  · intro q hq
    apply frame
    have sep:=p.allocation.weight
    unfold Physical.Separated at sep
    omega

#specNonCircular realTarget avoiding [KernelSource.kernel,KernelGrid.launch,realResult]
#specNonCircular nativeTarget avoiding [KernelSource.kernel,KernelGrid.launch,nativeResult,nativeHeap]
#axiomsClean native_initial
#axiomsClean real_correct
#axiomsClean native_correct
end
end VeriTile.Bench.Repairs.Conv2d.KernelEntries

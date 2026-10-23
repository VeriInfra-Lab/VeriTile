import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.InputBinding
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelRealIO

/-! Optional checked Tensor applications and Python admission correspondence.
The complete kernel specifications remain in Correct and FPEquiv. -/
namespace VeriTile.Bench.Repairs.KvCacheCopy.TensorBridge
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- Complete original static-loop kernel: successful execution, the independent
K/V tensor formulas for every dimension, and full exterior logical-memory frame. -/
theorem kv_cache_copy_real (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (s : BlockState) (binding : KernelMetadataBinding m b l t s)
    (hp : ValidProgram (config m b) s.pids) :
    Spec.Real (∃ final,exec (Source.originalKernel (parameters (config m b))).toAlgKernel s=some final ∧
      KernelCopy (config m b) l t s.pids s.readMem final.readMem ∧
      (∀ (r : RegionName) o,(∀ key,r≠outputRegion key ∨
        ∀ d,d<(config m b).dims → o≠outputIndex (config m b) l t key (s.pids 0) (s.pids 1) d) →
        final.mem r o=s.mem r o)) :=
  KernelReal.correctness m b l t h s binding hp

#axiomsClean kv_cache_copy_real

theorem python_checks_iff (m : Option Inputs) (b : Option Bool) (l : Lengths) (t : Tables) :
    checkArguments m b l t=.ok () ↔ Admitted m b l t := checkArguments_iff m b l t

/-- Actual Tensor data pointers supply only initial memory. No output formula
or wrapper execution is assumed by the caller binding. -/
theorem tensor_physical_correct (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (metadata : Memory Word) (payload : Nat → ℝ)
    (scratch : BlockState) (binding : TensorBinding m b l t id metadata)
    (hp : ValidProgram (config m b) scratch.pids) :
    Spec.Real (∃ final,
      exec (Source.originalKernel (parameters (config m b))).toAlgKernel
        (logicalState m metadata payload scratch)=some final ∧
      (∀ key d,d<(config m b).dims →
        final.readMem (outputRegion key) (outputIndex (config m b) l t key (scratch.pids 0) (scratch.pids 1) d)=
          payload (address m (inputRole key) (inputIndex (config m b) (scratch.pids 0) (scratch.pids 1) d))) ∧
      (∀ (r : RegionName) o,(∀ key,r≠outputRegion key ∨
        ∀ d,d<(config m b).dims → o≠outputIndex (config m b) l t key (scratch.pids 0) (scratch.pids 1) d) →
        final.mem r o=(logicalState m metadata payload scratch).mem r o)) := by
  obtain ⟨final,he,hv,hf⟩ := KernelReal.correctness m b l t h
    (logicalState m metadata payload scratch) (logicalState_binding m b l t metadata payload scratch binding) hp
  refine ⟨final,he,?_,hf⟩
  intro key d hd
  simpa only [(logicalState_payload m metadata payload scratch key _).1] using hv key d hd


/-- The admitted Tensor logical inputs are read from the very same physical
payload heap used to start this complete kernel. The read law concerns inputs only. -/
theorem tensor_input_correct {β : Type} (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (metadata : Memory Word) (heap : Memory β)
    (read : β → ℝ) (logical : Bool → Nat → ℝ) (reads : InputReadLaw m heap read logical)
    (scratch : BlockState) (binding : TensorBinding m b l t id metadata)
    (hp : ValidProgram (config m b) scratch.pids) :
    Spec.Real (∃ final,
      exec (Source.originalKernel (parameters (config m b))).toAlgKernel
        (logicalState m metadata (fun a => read (heap a)) scratch)=some final ∧
      (∀ key d,d<(config m b).dims →
        final.readMem (outputRegion key) (outputIndex (config m b) l t key (scratch.pids 0) (scratch.pids 1) d)=
          logical key (inputIndex (config m b) (scratch.pids 0) (scratch.pids 1) d)) ∧
      (∀ (r : RegionName) o,(∀ key,r≠outputRegion key ∨
        ∀ d,d<(config m b).dims → o≠outputIndex (config m b) l t key (scratch.pids 0) (scratch.pids 1) d) →
        final.mem r o=(logicalState m metadata (fun a => read (heap a)) scratch).mem r o)) := by
  obtain ⟨final,he,hv,hf⟩ := tensor_physical_correct m b l t h metadata (fun a => read (heap a)) scratch binding hp
  refine ⟨final,he,?_,hf⟩
  intro key d hd
  exact (hv key d hd).trans (reads.reads key _ (input_index_bound m b l t h scratch.pids hp key d hd))

/-- Both actual kernels start from the admitted Tensor payloads in one physical
heap. Their complete snapshot result/frame and the preserved logical K/V inputs
use the same input-only read law as the mathematical Tensor corollary. -/
theorem tensor_kernel_results {α β : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (_h : InputContract m b l t)
    (metadata : Memory Word) (heap : Memory β) (read : β → α) (logical : Bool → Nat → α)
    (reads : InputReadLaw m heap read logical) (scratch : FP.Structural.State α)
    (_binding : TensorBinding m b l t id metadata)
    (_hp : ValidProgram (config m b) scratch.pids) :
    let p := parameters (config m b)
    let initial := logicalFPState m metadata (fun a => read (heap a)) scratch
    ∃ lhs rhs,
      FP.Structural.exec M (Source.originalKernel p) initial=some lhs ∧
      FP.Structural.exec M (Source.commutedKernel p) initial=some rhs ∧
      lhs.mem=(KernelFP.copied M p initial (p.dims/p.width)).mem ∧ rhs.mem=lhs.mem ∧
      (∀ r o,(∀ key q,q<p.dims/p.width → r≠KernelFP.outputRegion key ∨
        ∀ i : TileIndex [p.width],o≠(KernelFP.targetInt p initial key q i).toNat) →
        lhs.mem r o=initial.mem r o) ∧
      (∀ key i,i<(m (inputRole key)).numel →
        (lhs.mem (inputRegion key) i).read .real=logical key i ∧
        (rhs.mem (inputRegion key) i).read .real=logical key i) := by
  dsimp only
  let p := parameters (config m b)
  let initial := logicalFPState m metadata (fun a => read (heap a)) scratch
  obtain ⟨lhs,rhs,hl,hr,hm,hn,hf⟩ := KernelNativeResults.kernel_results M p initial
  refine ⟨lhs,rhs,hl,hr,hm,hn,hf,?_⟩
  intro key i hi
  have outside : ∀ other q,q<p.dims/p.width → inputRegion key≠KernelFP.outputRegion other ∨
      ∀ j : TileIndex [p.width],i≠(KernelFP.targetInt p initial other q j).toNat := by
    intro other q hq
    left
    cases key <;> cases other <;> decide
  have keep := hf (inputRegion key) i outside
  have input := logical_fp_input m metadata heap read logical reads scratch key i hi
  constructor
  · rw [keep]
    exact input
  · rw [hn,keep]
    exact input

#axiomsClean python_checks_iff
#axiomsClean tensor_physical_correct
#axiomsClean tensor_input_correct
#axiomsClean tensor_kernel_results
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy.TensorBridge

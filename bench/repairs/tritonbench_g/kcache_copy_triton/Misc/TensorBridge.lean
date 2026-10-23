import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.KernelEquivalence
import bench.repairs.tritonbench_g.kcache_copy_triton.Correct
import bench.repairs.tritonbench_g.kcache_copy_triton.FPEquiv
import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.InputBinding
import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.KernelRealIO

/-! Optional checked Tensor applications and Python admission correspondence.
The complete source correctness and equivalence entries do not import this module. -/
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.TensorBridge
open VeriTile.Triton
noncomputable section

/-- The full shared source, including both dependent metadata loads and signed
multi-token indexing. Every selected component satisfies the independent tensor
formula, and every cell outside this program's destination slice is unchanged. -/
theorem kcache_copy_triton_real (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (s : BlockState)
    (binding : KernelMetadataBinding m n b l t s)
    (hp : ValidTile (config m n b) (programTile s.pids)) :
    Spec.Real (∃ final,
      exec (Source.originalKernel (parameters (config m n b))).toAlgKernel s=some final ∧
      KernelCopy (config m n b) l t s.pids (s.readMem "key") (final.readMem "cache") ∧
      (∀ (r : RegionName) o,r≠"cache" ∨ (∀ c,c<(config m n b).width →
        o≠outputIndex (config m n b) l t (s.pids 0) (s.pids 1) (s.pids 2*(config m n b).width+c)) →
        final.mem r o=s.mem r o)) :=
  KernelReal.correctness m n b l t h s binding hp



theorem python_checks_iff (m : Option Inputs) (n : Option Int) (b : Option Bool) (l : Lengths) (t : Tables) :
    checkArguments m n b l t=.ok () ↔ Admitted m n b l t := checkArguments_iff m n b l t

/-- The same theorem instantiated at the actual caller data pointers. This is
an initial-memory binding; it does not assume any output or wrapper execution. -/
theorem tensor_physical_correct (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (metadata : Memory Word) (payload : Nat → ℝ)
    (scratch : BlockState) (binding : TensorBinding m n b l t id metadata)
    (hp : ValidTile (config m n b) (programTile scratch.pids)) :
    Spec.Real (∃ final,
      exec (Source.originalKernel (parameters (config m n b))).toAlgKernel
        (logicalState m metadata payload scratch)=some final ∧
      KernelCopy (config m n b) l t scratch.pids
        (fun i => payload (address m .key i)) (final.readMem "cache") ∧
      (∀ (r : RegionName) o,r≠"cache" ∨ (∀ c,c<(config m n b).width →
        o≠outputIndex (config m n b) l t (scratch.pids 0) (scratch.pids 1)
          (scratch.pids 2*(config m n b).width+c)) →
        final.mem r o=(logicalState m metadata payload scratch).mem r o)) := by
  obtain ⟨final,he,hv,hf⟩ := KernelReal.correctness m n b l t h
    (logicalState m metadata payload scratch) (logicalState_binding m n b l t metadata payload scratch binding) hp
  refine ⟨final,he,?_,hf⟩
  intro c hc
  simpa only [(logicalState_payload m metadata payload scratch _).1] using hv c hc


/-- The mathematical output copies the admitted K Tensor's logical values,
using one input-only read law for this same physical payload heap. -/
theorem tensor_input_correct {β : Type} (m : Inputs) (n : Nat) (b : Bool)
    (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (metadata : Memory Word) (heap : Memory β) (read : β → ℝ) (logical : Nat → ℝ)
    (reads : InputReadLaw m heap read logical) (scratch : BlockState)
    (binding : TensorBinding m n b l t id metadata)
    (hp : ValidTile (config m n b) (programTile scratch.pids)) :
    Spec.Real (∃ final,
      exec (Source.originalKernel (parameters (config m n b))).toAlgKernel
        (logicalState m metadata (fun a => read (heap a)) scratch)=some final ∧
      KernelCopy (config m n b) l t scratch.pids logical (final.readMem "cache") ∧
      (∀ (r : RegionName) o,r≠"cache" ∨ (∀ c,c<(config m n b).width →
        o≠outputIndex (config m n b) l t (scratch.pids 0) (scratch.pids 1)
          (scratch.pids 2*(config m n b).width+c)) →
        final.mem r o=(logicalState m metadata (fun a => read (heap a)) scratch).mem r o)) := by
  obtain ⟨final,he,hv,hf⟩ := tensor_physical_correct m n b l t h metadata
    (fun a => read (heap a)) scratch binding hp
  refine ⟨final,he,?_,hf⟩
  intro c hc
  exact (hv c hc).trans (reads.reads _ (input_index_bound m n b l t h scratch.pids hp c hc))

/-- The complete FP kernels execute from the same admitted K Tensor and
physical payload heap used by its logical input law, retaining the full
snapshot result/frame and the logical input cells on both sides. -/
theorem tensor_kernel_results {α β : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (_h : InputContract m n b l t)
    (metadata : Memory Word) (heap : Memory β) (read : β → α) (logical : Nat → α)
    (reads : InputReadLaw m heap read logical) (scratch : FP.Structural.State α)
    (_binding : TensorBinding m n b l t id metadata)
    (_hp : ValidTile (config m n b) (programTile scratch.pids)) :
    let p := parameters (config m n b)
    let initial := logicalFPState m metadata (fun a => read (heap a)) scratch
    ∃ lhs rhs,
      FP.Structural.exec M (Source.originalKernel p) initial=some lhs ∧
      FP.Structural.exec M (Source.commutedKernel p) initial=some rhs ∧
      lhs.mem=(KernelFP.copied M p initial).mem ∧ rhs.mem=lhs.mem ∧
      (∀ r o,r≠p.KCache ∨ (∀ i : TileIndex [p.width],o≠KernelFP.targetOffset p initial i) →
        lhs.mem r o=initial.mem r o) ∧
      (∀ i,i<(m .key).numel → (lhs.mem "key" i).read .real=logical i ∧
        (rhs.mem "key" i).read .real=logical i) := by
  dsimp only
  let p := parameters (config m n b)
  let initial := logicalFPState m metadata (fun a => read (heap a)) scratch
  obtain ⟨lhs,rhs,hl,hr,hm,hn,hf⟩ := KernelEquivalence.kernel_results M p initial
  refine ⟨lhs,rhs,hl,hr,hm,hn,hf,?_⟩
  intro i hi
  have keep := hf "key" i (Or.inl (by change ("key" : RegionName) ≠ "cache"; decide))
  have input := logical_fp_input m metadata heap read logical reads scratch i hi
  constructor
  · rw [keep]
    exact input
  · rw [hn,keep]
    exact input

#axiomsClean kcache_copy_triton_real
#axiomsClean python_checks_iff
#axiomsClean tensor_physical_correct
#axiomsClean tensor_input_correct
#axiomsClean tensor_kernel_results
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.TensorBridge

import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Host
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.AdmissionCore
import bench.repairs.tritonbench_g.cache_transform_triton.Contract
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Entry
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Schedule

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural

/-- Logical views describe the admitted caller tensors, not copied buffers.
Their input byte/coordinate relation is the ordinary contiguous view law. -/
structure ViewBinding {α : Type} (r : Request) (cos sin : LogicalView α) (width : Nat) : Prop where
  cosRows : cos.rows=r.cosRows
  sinRows : sin.rows=r.sinRows
  cosWidth : cos.width=r.width
  sinWidth : sin.width=r.width
  width_eq : width=r.width
  cosDevice : cos.device=r.cos.device.index
  sinDevice : sin.device=r.sin.device.index

structure Returned (α : Type) where
  state : State α
  (cos sin : OutputDescriptor)
  restoredDevice : Nat

/-- Complete successful wrapper return: the two newly allocated Tensor
objects, their exact source-executed state, and restored device context. -/
noncomputable def returned {α β : Type} [Inhabited α] (M : Algebra α)
    (cos sin : Parameters) (s : State α) (r : Request) (cv sv : LogicalView β)
    (caller cosAllocation sinAllocation : Nat) : Returned α :=
  ⟨pairGather M cos sin s,allocatedOutput cv r.mode r.naturalLengths cosAllocation,
    allocatedOutput sv r.mode r.naturalLengths sinAllocation,caller⟩

theorem returned_contract {α β : Type} [Inhabited α] (M : Algebra α)
    (cos sin : Parameters) (s : State α) (r : Request) (cv sv : LogicalView β)
    (width : Nat) (binding : ViewBinding r cv sv width) (caller ca sa : Nat) :
    let result := returned M cos sin s r cv sv caller ca sa
    result.cos.rows=r.outputRows ∧ result.sin.rows=r.outputRows ∧
    result.cos.width=r.width ∧ result.sin.width=r.width ∧
    result.cos.dtype=.fp32 ∧ result.sin.dtype=.fp32 ∧
    result.cos.device=r.cos.device.index ∧ result.sin.device=r.sin.device.index ∧
    result.cos.contiguous=Bool.true ∧ result.sin.contiguous=Bool.true ∧
    result.cos.requiresGrad=Bool.false ∧ result.sin.requiresGrad=Bool.false ∧
    result.cos.allocation=ca ∧ result.sin.allocation=sa ∧ result.restoredDevice=caller := by
  cases dc : cv.dtype
  cases ds : sv.dtype
  simp [returned,allocatedOutput,←outputRows_eq,binding.cosWidth,binding.sinWidth,
    binding.cosDevice,binding.sinDevice,dc,ds]

theorem entry_single_launch {α β : Type} [Inhabited α] (r : Request) (h : InputContract r)
    (p : Parameters) (s : State α) (v : LogicalView β) (width : Nat)
    (observe : (Fin v.dtype.bytes → Nat) → β) (view : DirectView v width observe)
    (entry : EntryBinding p s r.mode r.naturalLengths v width observe view)
    (hw : width=r.width) (k : Nat) (hk : k<blocks p.count) :
    (tileParameters p k).start=0 ∧ localPid k=k := by
  have count : p.count=r.count := by
    rw [entry.count,←outputRows_eq,hw]
    cases v.dtype
    simp [Request.count,Storage.bytes,Nat.mul_assoc]
  simpa only [tileParameters,count] using single_launch r h k (count ▸ hk)

#axiomsClean checkInputs_iff
#axiomsClean admitted_lengths
#axiomsClean admitted_signed
#axiomsClean single_launch
#axiomsClean entry_single_launch
#axiomsClean returned_contract
end VeriTile.Bench.Repairs.CacheTransform

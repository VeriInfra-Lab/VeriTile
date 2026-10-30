import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.KernelContract
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
open VeriTile.Triton
noncomputable section

/-- Ordinary input-only logical read of the admitted K Tensor from the current
physical payload heap. This premise mentions no kernel, output or numerical law. -/
structure InputReadLaw {β γ : Type} (m : Inputs) (heap : Memory β) (read : β → γ)
    (logical : Nat → γ) : Prop where
  reads : ∀ i,i<(m .key).numel → read (heap (address m .key i))=logical i

theorem input_index_bound (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (ids : Nat → Nat)
    (hp : ValidTile (config m n b) (programTile ids)) (c : Nat) (hc : c<(config m n b).width) :
    inputIndex (config m n b) (ids 0) (ids 1) (ids 2*(config m n b).width+c)<(m .key).numel := by
  rw [(numel m n b l t h).1]
  have dim := component_bound (config m n b) (split_geometry m n b l t h).2 ⟨programTile ids,c⟩ ⟨hp,hc⟩
  exact Linear.linear_bound _ _ _ _ (Linear.linear_bound _ _ _ _ hp.1 hp.2.1) dim

theorem logical_real_input {β : Type} (m : Inputs) (metadata : Memory Word)
    (heap : Memory β) (read : β → ℝ) (logical : Nat → ℝ)
    (law : InputReadLaw m heap read logical) (scratch : BlockState)
    (i : Nat) (hi : i<(m .key).numel) :
    (logicalState m metadata (fun a => read (heap a)) scratch).readMem "key" i=logical i :=
  (logicalState_payload m metadata _ scratch i).1.trans (law.reads i hi)

/-- Initial FP memory uses the same actual data pointers and shared payload
heap as the logical read law; it assumes no result of execution. -/
def logicalFPState {α : Type} (m : Inputs) (metadata : Memory Word) (payload : Nat → α)
    (scratch : FP.Structural.State α) : FP.Structural.State α :=
  { scratch with mem := fun r i =>
      if r="lengths" then .mk .int (metadata (address m .lengths i)).toInt
      else if r="table" then .mk .int (metadata (address m .table i)).toInt
      else if r="key" then .mk .real (payload (address m .key i))
      else if r="cache" then .mk .real (payload (address m .cache i))
      else scratch.mem r i }

theorem logical_fp_input {α β : Type} [Inhabited α] (m : Inputs) (metadata : Memory Word)
    (heap : Memory β) (read : β → α) (logical : Nat → α)
    (law : InputReadLaw m heap read logical) (scratch : FP.Structural.State α)
    (i : Nat) (hi : i<(m .key).numel) :
    ((logicalFPState m metadata (fun a => read (heap a)) scratch).mem "key" i).read .real=logical i := by
  simpa [logicalFPState] using law.reads i hi

#axiomsClean input_index_bound
#axiomsClean logical_real_input
#axiomsClean logical_fp_input
#specNonCircular InputReadLaw avoiding [logicalState,logicalFPState,Source.originalKernel,Source.commutedKernel]
end
end VeriTile.Bench.Repairs.KcacheCopyTriton

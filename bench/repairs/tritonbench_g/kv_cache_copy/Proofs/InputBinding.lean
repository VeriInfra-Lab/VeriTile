import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelContract
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.KvCacheCopy
open VeriTile.Triton
noncomputable section

/-- Ordinary input-only Tensor read law. The same admitted K/V metadata and
one physical payload heap determine both logical arrays. No kernel, output,
allocation or numerical identity occurs in this representation premise. -/
structure InputReadLaw {β γ : Type} (m : Inputs) (heap : Memory β) (read : β → γ)
    (logical : Bool → Nat → γ) : Prop where
  reads : ∀ key i,i<(m (inputRole key)).numel →
    read (heap (address m (inputRole key) i))=logical key i

theorem input_index_bound (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (ids : Nat → Nat) (hp : ValidProgram (config m b) ids)
    (key : Bool) (d : Nat) (hd : d<(config m b).dims) :
    inputIndex (config m b) (ids 0) (ids 1) d<(m (inputRole key)).numel := by
  have sizes := numel m b l t h
  have bound := Linear.linear_bound _ _ _ _ (Linear.linear_bound _ _ _ _ hp.1 hp.2) hd
  cases key <;> simpa only [inputRole,Bool.false_eq_true,ite_false,ite_true,
    sizes.1,sizes.2.1,inputIndex] using bound

theorem logical_real_input {β : Type} (m : Inputs) (metadata : Memory Word)
    (heap : Memory β) (read : β → ℝ) (logical : Bool → Nat → ℝ)
    (law : InputReadLaw m heap read logical) (scratch : BlockState)
    (key : Bool) (i : Nat) (hi : i<(m (inputRole key)).numel) :
    (logicalState m metadata (fun a => read (heap a)) scratch).readMem (inputRegion key) i=logical key i :=
  (logicalState_payload m metadata _ scratch key i).1.trans (law.reads key i hi)

/-- FP view of the same current physical input/cache payload heap. Only the
representation changes; this definition performs no execution or output write. -/
def logicalFPState {α : Type} (m : Inputs) (metadata : Memory Word) (payload : Nat → α)
    (scratch : FP.Structural.State α) : FP.Structural.State α :=
  { scratch with mem := fun r i =>
      if r="lengths" then .mk .int (metadata (address m .lengths i)).toInt
      else if r="table" then .mk .int (metadata (address m .table i)).toInt
      else if r="key" then .mk .real (payload (address m .key i))
      else if r="value" then .mk .real (payload (address m .value i))
      else if r="keyCache" then .mk .real (payload (address m .keyCache i))
      else if r="valueCache" then .mk .real (payload (address m .valueCache i))
      else scratch.mem r i }

theorem logical_fp_input {α β : Type} [Inhabited α] (m : Inputs) (metadata : Memory Word)
    (heap : Memory β) (read : β → α) (logical : Bool → Nat → α)
    (law : InputReadLaw m heap read logical) (scratch : FP.Structural.State α)
    (key : Bool) (i : Nat) (hi : i<(m (inputRole key)).numel) :
    ((logicalFPState m metadata (fun a => read (heap a)) scratch).mem (inputRegion key) i).read .real=logical key i := by
  have bound := law.reads key i hi
  cases key <;> simpa [logicalFPState,inputRegion,inputRole] using bound

#axiomsClean input_index_bound
#axiomsClean logical_real_input
#axiomsClean logical_fp_input
#specNonCircular InputReadLaw avoiding [logicalState,logicalFPState,Source.originalKernel,Source.commutedKernel]
end
end VeriTile.Bench.Repairs.KvCacheCopy

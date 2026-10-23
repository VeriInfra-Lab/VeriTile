import bench.repairs.tritonbench_g.quantize_global.Proofs.Maximum
import bench.repairs.tritonbench_g.quantize_global.Proofs.Execution
namespace QuantizeGlobalRepair
open VeriTile.Triton FP.Structural
noncomputable section

structure View where
  shape : List Nat
  strides : List Nat
  storageOffset : Nat
  storageLength : Nat
  negative : Bool
  device : Nat
  original : RegionName
def View.count (v : View) := v.shape.prod
def View.offsets (v : View) := QuantizeGlobalFinite.viewOffsets v.shape v.strides
def View.Valid (v : View) : Prop := v.shape.length = v.strides.length ∧
  ∀ i ∈ v.offsets, v.storageOffset + i < v.storageLength
def View.logical [Inhabited α] (v : View) (neg : α → α) (s : State α) (i : Nat) : α :=
  let x := (s.mem v.original (v.storageOffset + v.offsets[i]!)).read .real
  if v.negative then neg x else x

theorem view_length (shape strides : List Nat) (h : shape.length = strides.length) :
    (QuantizeGlobalFinite.viewOffsets shape strides).length = shape.prod := by
  induction shape generalizing strides with
  | nil => cases strides <;> simp_all [QuantizeGlobalFinite.viewOffsets]
  | cons n ns ih =>
    cases strides with
    | nil => simp at h
    | cons t ts =>
      have hh : ns.length = ts.length := by simpa using h
      simp [QuantizeGlobalFinite.viewOffsets,List.length_flatMap,ih ts hh]

structure Allocation where
  snapshot : RegionName
  maximum : RegionName
  output : RegionName
  initialOutput : Nat → BitVec 8
def Allocation.Fresh (a : Allocation) (v : View) : Prop :=
  a.snapshot ≠ a.maximum ∧ a.snapshot ≠ a.output ∧ a.maximum ≠ a.output ∧
  v.original ≠ a.snapshot ∧ v.original ≠ a.maximum ∧ v.original ≠ a.output
def parameters (v : View) (a : Allocation) : Parameters :=
  ⟨a.snapshot,a.maximum,a.output,v.count⟩

/-- Ordinary resolve_neg/contiguous clone and fresh allocation semantics.
Only the finite windows owned by these allocations are written. -/
def prepare (v : View) (alloc : Allocation) (xs : Nat → α) (maximum : α)
    (s : State α) : State α :=
  { s with mem := fun r i =>
      if r = alloc.snapshot ∧ i < v.count then .mk .real (xs i)
      else if r = alloc.maximum ∧ i = 0 then .mk .real maximum
      else if r = alloc.output ∧ i < v.count then .mk .int (alloc.initialOutput i).toInt
      else s.mem r i }

structure Returned (α : Type) where
  state : State α
  outputShape : List Nat
  maximumShape : List Nat
  outputDevice : Nat
  maximumDevice : Nat
  output : RegionName
  maximum : RegionName

def hostRun [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (absolute : α → α) (maximum : α → α → α) (tree : MaxTree)
    (v : View) (alloc : Allocation) (trials : Nat → List Config) (s : State α) : Option (Returned α) := do
  let xs := v.logical a.neg s
  let m := tree.eval (a.literal 0) absolute maximum xs
  let t ← execute rest a (parameters v alloc) (schedule v.count trials) (prepare v alloc xs m s)
  pure ⟨t,v.shape,[1],v.device,v.device,alloc.output,alloc.maximum⟩

theorem prepared_inputs [Inhabited α] (v : View) (alloc : Allocation)
    (xs : Nat → α) (m : α) (s : State α) (fresh : alloc.Fresh v) :
    Inputs (parameters v alloc) xs m (prepare v alloc xs m s) := by
  constructor
  · intro i hi
    change i < v.count at hi
    simp [prepare,parameters,hi]
  · simp [prepare,parameters,Ne.symm fresh.1]

theorem host_execution [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (absolute : α → α) (maximum : α → α → α) (tree : MaxTree)
    (v : View) (alloc : Allocation) (trials : Nat → List Config) (s : State α)
    (fresh : alloc.Fresh v) (nonempty : ∀ k, k < segmentCount v.count → trials k ≠ []) :
    ∃ ret, hostRun rest a absolute maximum tree v alloc trials s = some ret ∧
      ret.outputShape = v.shape ∧ ret.maximumShape = [1] ∧
      ret.outputDevice = v.device ∧ ret.maximumDevice = v.device ∧
      ret.output = alloc.output ∧ ret.maximum = alloc.maximum ∧
      let xs := v.logical a.neg s
      let m := tree.eval (a.literal 0) absolute maximum xs
      ret.state.mem = complete (parameters v alloc) (fun i => scalar a (xs i) m)
        (prepare v alloc xs m s).mem := by
  obtain ⟨t,ht,hm⟩ := whole_grid rest a (parameters v alloc) (v.logical a.neg s)
    (tree.eval (a.literal 0) absolute maximum (v.logical a.neg s)) trials nonempty
    ⟨fresh.2.1,fresh.2.2.1⟩ _ (prepared_inputs v alloc _ _ s fresh)
  change execute rest a (parameters v alloc) (schedule v.count trials)
    (prepare v alloc (v.logical a.neg s)
      (tree.eval (a.literal 0) absolute maximum (v.logical a.neg s)) s) = some t at ht
  exact ⟨⟨t,v.shape,[1],v.device,v.device,alloc.output,alloc.maximum⟩,
    by simp only [hostRun,ht,Option.bind_some]; rfl,
    rfl,rfl,rfl,rfl,rfl,rfl,hm⟩

theorem full_frame (v : View) (alloc : Allocation) (xs : Nat → α) (m : α)
    (values : Nat → BitVec 8) (s : State α) (r : RegionName) (i : Nat)
    (hx : ¬ (r = alloc.snapshot ∧ i < v.count))
    (hm : ¬ (r = alloc.maximum ∧ i = 0))
    (ho : ¬ (r = alloc.output ∧ i < v.count)) :
    complete (parameters v alloc) values (prepare v alloc xs m s).mem r i = s.mem r i := by
  simp [complete,parameters,prepare,hx,hm,ho]

theorem original_preserved (v : View) (alloc : Allocation) (xs : Nat → α) (m : α)
    (values : Nat → BitVec 8) (s : State α) (fresh : alloc.Fresh v) (i : Nat) :
    complete (parameters v alloc) values (prepare v alloc xs m s).mem v.original i = s.mem v.original i := by
  apply full_frame <;> simp [fresh.2.2.2.1,fresh.2.2.2.2.1,fresh.2.2.2.2.2]

#axiomsClean host_execution
#axiomsClean original_preserved
#axiomsClean full_frame
end
end QuantizeGlobalRepair

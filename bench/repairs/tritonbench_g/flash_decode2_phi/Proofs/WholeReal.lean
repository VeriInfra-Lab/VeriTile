import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.NativeSafety
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.RealResult

namespace VeriTile.Bench.Repairs.FlashDecode2.WholeReal
open VeriTile.Triton FP.Structural Checked TensorBinding
noncomputable section

abbrev Heap := RegionName → Nat → ℝ
abbrev Memory := RegionName → Nat → Option ℝ

def entry (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap) : State (Option ℝ) where
  mem := fun r i => if r = "L" then .mk .nat (lengthValues t memory i).toNat
    else .mk .real (some (heap (t.names.at r) i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def prepare (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap)
    (h : InputContract c (lengthValues t memory)) :=
  preparation c t memory h (entry t memory heap) (fun _ => rfl)

/-- Ordinary allocation-name transport of the typed source output view.
The source performs all writes; this operation changes only the view's name. -/
def physical (t : Tensors c allocations) (heap : Heap) (state : State (Option ℝ)) : Memory :=
  fun r i => if r = t.names.out then (state.mem "O" i).read .real else some (heap r i)

def reference (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap)
    (row d : Nat) : ℝ :=
  Reference.target (fun j => heap t.names.log (row*capacity c+j))
    (fun j d => heap t.names.mid (row*capacity c*width c+j*width c+d))
    (((lengthValues t memory (row/heads c)).toNat+c.blockSeq-1)/c.blockSeq) d

structure Result where
  returned : Option Unit
  output : VeriTile.Torch.FP32.Tensor
  memory : Memory
  currentDevice : Nat
  launchDevice : Option Nat

def invoke (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap) :
    Except String Result :=
  if admitted : checkInput c (lengthValues t memory) = .ok () then
    let h := (check_input_iff c _).mp admitted
    let call := prepare c t memory heap h
    match Host.invoke Reference.ideal (request c h) call (List.range (geometry c).rows) with
    | none => .error "source execution"
    | some result => .ok ⟨result.returned,t.output.tensor,physical t heap result.state,
        result.currentDevice,result.launchDevice⟩
  else .error "input contract rejected"

theorem finite (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap) :
    FiniteInputs (entry t memory heap) (geometry c) := by
  constructor
  · intro row hr j hj
    exact ⟨_,rfl⟩
  · intro row hr j hj d hd
    exact ⟨_,rfl⟩

theorem formula_reference (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap)
    (h : InputContract c (lengthValues t memory)) (row d : Nat) (hr : row < (geometry c).rows) :
    formula (entry t memory heap) (geometry c) row d = reference c t memory heap row d := by
  let call := prepare c t memory heap h
  obtain ⟨positiveHeads,batch,head⟩ := Host.row_indices (request c h) row hr
  have length := (call.lengthValues (row/heads c) batch positiveHeads).1
  have countLaw := count_eq (Grid.setup (batches c) (heads c) row (entry t memory heap)) c.blockSeq
    (lengthValues t memory (row/heads c)).toNat rfl length
  unfold formula
  change Reference.target _ _ (count (Grid.setup (batches c) (heads c) row (entry t memory heap)) c.blockSeq) d = _
  rw [countLaw]
  rfl

structure Guarantee (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap)
    (result : Result) : Prop where
  returned : result.returned = none
  output : result.output = t.output.tensor
  storage : allocations t.names.out = result.output.storage
  shape : result.output.metadata.shape = [batches c,heads c,width c]
  dtype : result.output.metadata.dtype = .floating .fp32
  context : result.currentDevice = t.callerDevice
  launch : result.launchDevice = if (geometry c).rows = 0 then none else some c.mid.metadata.device.index
  addresses : NativeSafety.AllAccess c
  coordinates : NativeSafety.Coordinates c t
  arithmetic : NativeSafety.Arithmetic c t memory
  values : ∀ row (k : TileIndex [block c]),row < (geometry c).rows → lane k < width c →
    result.memory t.names.out (row*width c+lane k) = some (reference c t memory heap row (lane k))
  frame : ∀ r i,r ≠ t.names.out ∨ (geometry c).size ≤ i → result.memory r i = some (heap r i)

theorem tensor (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap)
    (h : InputContract c (lengthValues t memory)) :
    ∃ result,invoke c t memory heap = .ok result ∧ Guarantee c t memory heap result := by
  let call := prepare c t memory heap h
  obtain ⟨result,run,ret,out,context,launch,values,frame⟩ :=
    FlashDecode2.correct (request c h) call (finite c t memory heap)
      (List.range (geometry c).rows) (List.Perm.refl _)
  let answer : Result := ⟨result.returned,t.output.tensor,physical t heap result.state,
    result.currentDevice,result.launchDevice⟩
  refine ⟨answer,?_,ret,rfl,t.outputStorage,?_,t.output.laws.dtype,context,launch,
    NativeSafety.checked_access c t memory h,NativeSafety.checked_coordinates c t h.metadata,
    NativeSafety.checked_arithmetic c t memory h,?_,?_⟩
  · simp only [invoke,dif_pos ((check_input_iff c _).mpr h)]
    rw [run]
  · have shape := h.metadata.2.2.2.2.2.2.1
    rw [←t.outputObserved] at shape
    exact shape
  · intro row k hr hk
    change (physical t heap result.state) t.names.out (row*width c+lane k) = _
    have value := values row k hr hk
    change result.state.mem "O" (row*width c+lane k) =
      .mk .real (some (formula (entry t memory heap) (geometry c) row (lane k))) at value
    simp only [physical,ite_true,value,Cell.read_same]
    rw [formula_reference c t memory heap h row (lane k) hr]
  · intro r i outside
    change physical t heap result.state r i = _
    by_cases same : r = t.names.out
    · subst r
      have beyond : (geometry c).size ≤ i := outside.resolve_left (not_not.mpr rfl)
      simp only [physical,ite_true,frame "O" i (.inr beyond)]
      rfl
    · simp [physical,same]

theorem empty (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) (heap : Heap)
    (h : InputContract c (lengthValues t memory)) (empty : batches c = 0 ∨ heads c = 0) :
    invoke c t memory heap = .ok ⟨none,t.output.tensor,fun r i => some (heap r i),t.callerDevice,none⟩ := by
  have source := Host.empty_return Reference.ideal (request c h) (prepare c t memory heap h)
    (List.range (geometry c).rows) (List.Perm.refl _) empty
  simp only [invoke,dif_pos ((check_input_iff c _).mpr h),source]
  have transport : physical t heap (entry t memory heap) = fun r i => some (heap r i) := by
    funext r i
    by_cases same : r = t.names.out
    · subst r; simp [physical,entry,Cell.read_same,Native.Bindings.at]
    · simp [physical,same]
  change Except.ok (Result.mk none t.output.tensor (physical t heap (entry t memory heap))
    t.callerDevice none) = _
  rw [transport]

#axiomsClean tensor
#axiomsClean empty
#specNonCircular reference avoiding [invoke,Host.invoke,Grid.source,values]
end
end VeriTile.Bench.Repairs.FlashDecode2.WholeReal

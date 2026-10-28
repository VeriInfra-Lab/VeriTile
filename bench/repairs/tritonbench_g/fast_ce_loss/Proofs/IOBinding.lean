import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.IOValues

namespace VeriTile.Bench.Repairs.FastCeLoss.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

@[simp] theorem forward_row (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (isChunked : Bool) (s : State α) :
    ((forwardInvocation M p row chunk isChunked).initial s).pids 0=row := by
  simp [forwardInvocation,Invocation.initial,bindRegisters]
@[simp] theorem chunked_chunk (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (s : State α) :
    ((forwardInvocation M p row chunk Bool.true).initial s).pids 1=chunk := by
  simp [forwardInvocation,Invocation.initial,bindRegisters]
@[simp] theorem backward_row (M : Algebra α) (p : Parameters) (row chunk : Nat) (s : State α) :
    ((backwardInvocation M p row chunk).initial s).pids 0=row := by
  simp [backwardInvocation,Invocation.initial,bindRegisters]
@[simp] theorem backward_chunk (M : Algebra α) (p : Parameters) (row chunk : Nat) (s : State α) :
    ((backwardInvocation M p row chunk).initial s).pids 1=chunk := by
  simp [backwardInvocation,Invocation.initial,bindRegisters]

theorem forward_label (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (isChunked wholeRow : Bool) (xs : ChannelValues α (forwardInputs p wholeRow)) (s : State α)
    (bound : (forwardIO M id id p row chunk isChunked wholeRow).InputBinding xs s) :
    label p ((forwardInvocation M p row chunk isChunked).initial s)=forwardLabel p wholeRow xs := by
  simpa [label,forwardLabel,forwardIO,forwardInputs,forwardRead,mapValue] using
    bound (0 : Fin 3) PUnit.unit trivial

theorem forward_tile (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (isChunked wholeRow : Bool) (xs : ChannelValues α (forwardInputs p wholeRow)) (s : State α)
    (bound : (forwardIO M id id p row chunk isChunked wholeRow).InputBinding xs s)
    (lane : Fin p.block) (active : chunk*p.block+lane.val<p.vocab) :
    (((forwardInvocation M p row chunk isChunked).initial s).mem p.logits
      (row*p.logitsStride+(chunk*p.block+lane.val))).read .real=
        tile p wholeRow chunk (xs (1 : Fin 3)) lane := by
  have hi := tile_index p wholeRow chunk lane active
  have h := bound (1 : Fin 3) (⟨_,hi⟩,PUnit.unit)
  cases wholeRow <;>
    simpa [forwardIO,forwardInputs,forwardRead,forwardInputMask,column,width,tile,mapValue,active,
      lane.isLt] using h (by simpa [forwardIO,forwardInputMask,column] using active)

theorem forward_gather (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (isChunked wholeRow : Bool) (xs : ChannelValues α (forwardInputs p wholeRow)) (s : State α)
    (bound : (forwardIO M id id p row chunk isChunked wholeRow).InputBinding xs s)
    (written : isChunked=Bool.false ∨ chunk=0) (active : forwardLabel p wholeRow xs ≠ -100) :
    (s.mem p.logits (row*p.logitsStride+(forwardLabel p wholeRow xs).toNat-(-(forwardLabel p wholeRow xs)).toNat)).read .real=
      xs (2 : Fin 3) PUnit.unit :=
  bound (2 : Fin 3) PUnit.unit ⟨written,active⟩

theorem backward_label (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (wholeRow : Bool) (xs : ChannelValues α (backwardInputs p wholeRow)) (s : State α)
    (bound : (backwardIO M id id p row chunk wholeRow).InputBinding xs s) :
    label p ((backwardInvocation M p row chunk).initial s)=backwardLabel p wholeRow xs := by
  simpa [label,backwardLabel,backwardIO,backwardInputs,backwardRead,mapValue] using
    bound (0 : Fin 4) PUnit.unit trivial

theorem backward_tile (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (wholeRow : Bool) (xs : ChannelValues α (backwardInputs p wholeRow)) (s : State α)
    (bound : (backwardIO M id id p row chunk wholeRow).InputBinding xs s)
    (lane : Fin p.block) (active : chunk*p.block+lane.val<p.vocab) :
    (s.mem p.logits (row*p.logitsStride+(chunk*p.block+lane.val))).read .real=
      tile p wholeRow chunk (xs (1 : Fin 4)) lane := by
  have hi := tile_index p wholeRow chunk lane active
  have h := bound (1 : Fin 4) (⟨_,hi⟩,PUnit.unit)
  cases wholeRow <;>
    simpa [backwardIO,backwardInputs,backwardRead,backwardInputMask,column,width,tile,mapValue,active,
      lane.isLt] using h (by simpa [backwardIO,backwardInputMask,column] using active)


theorem forward_tile_bounds (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (isChunked wholeRow : Bool) (xs : ChannelValues α (forwardInputs p wholeRow))
    (bounds : RegionName → Nat)
    (bounded : (forwardIO M id id p row chunk isChunked wholeRow).InBounds bounds xs)
    (lane : Fin p.block) (active : chunk*p.block+lane.val<p.vocab) :
    row*p.logitsStride+(chunk*p.block+lane.val)<bounds p.logits := by
  have hi := tile_index p wholeRow chunk lane active
  have h := bounded.2.1 (1 : Fin 3) (⟨_,hi⟩,PUnit.unit)
  cases wholeRow <;>
    simpa [forwardIO,forwardInputs,forwardRead,column,width] using
      h (by simpa [forwardIO,forwardInputMask,column] using active)

theorem backward_tile_bounds (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (wholeRow : Bool) (xs : ChannelValues α (backwardInputs p wholeRow))
    (bounds : RegionName → Nat)
    (bounded : (backwardIO M id id p row chunk wholeRow).InBounds bounds xs)
    (lane : Fin p.block) (active : chunk*p.block+lane.val<p.vocab) :
    row*p.logitsStride+(chunk*p.block+lane.val)<bounds p.logits := by
  have hi := tile_index p wholeRow chunk lane active
  have h := bounded.2.1 (1 : Fin 4) (⟨_,hi⟩,PUnit.unit)
  cases wholeRow <;>
    simpa [backwardIO,backwardInputs,backwardRead,column,width] using
      h (by simpa [backwardIO,backwardInputMask,column] using active)

end
end VeriTile.Bench.Repairs.FastCeLoss.PublicIO

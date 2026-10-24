import bench.repairs.tritonbench_g.fast_rms_layernorm.Contract
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostIdentity

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.CheckedHost
open VeriTile.Triton Host Contract Storage
noncomputable section
variable {α β : Type}

/-- Ordinary tensor-view identities, including pointers and backing identities.
The allocation metadata comes from input/allocator bindings, not a numerical
or kernel-success callback. The two forward input handles are direct views. -/
structure DirectViews (r : Request α) (m : Metadata) (h : Call) (live : Caller)
    (p : Prepared h .forward live) where
  originalX : Native.Allocation h.cells
  originalW : Native.Allocation h.cols
  inputId : originalX.storageId=m.input.storageId
  weightId : originalW.storageId=m.weight.storageId
  inputAddress : originalX.buffer.base=m.input.address
  weightAddress : originalW.buffer.base=m.weight.address
  inputView : p.storage.allocation "x" (by simp [buffers])=
    (show Native.Allocation (cells h "x") from originalX)
  weightView : p.storage.allocation "w" (by simp [buffers])=
    (show Native.Allocation (cells h "w") from originalW)
  inputRead : ∀ off,off<h.cells → ∀ byte : Fin 4,
    live (originalX.buffer.base+off*4+byte.val)
  weightRead : ∀ off,off<h.cols → ∀ byte : Fin 4,
    live (originalW.buffer.base+off*4+byte.val)

/-- Decoding of actual caller bytes at input entry; no result equation occurs. -/
def InputsAtEntry (r : Request α) (h : Call) (live : Caller) (p : Prepared h .forward live)
    (encode : α → Fin 4 → β) (entry : Heap β) : Prop :=
  (∀ off,off<h.cells → ∀ byte,
    entry (address p.storage "x" (by simp [buffers]) off byte)=encode (r.x off) byte) ∧
  (∀ off,off<h.cols → ∀ byte,
    entry (address p.storage "w" (by simp [buffers]) off byte)=encode (r.w off) byte)

theorem saved_input_identity (r : Request α) (m : Metadata) (h : Call) (live : Caller)
    (p : Prepared h .forward live) (views : DirectViews r m h live p) (actual : CellHeap α) :
    ((makeContext r h p actual).storage.allocation "x" (by simp [buffers])).storageId=m.input.storageId ∧
    ((makeContext r h p actual).storage.allocation "w" (by simp [buffers])).storageId=m.weight.storageId := by
  change (p.storage.allocation "x" _).storageId=_ ∧ (p.storage.allocation "w" _).storageId=_
  rw [views.inputView,views.weightView]
  exact ⟨views.inputId,views.weightId⟩

/-- Public checks precede all direct views and output allocations. -/
def forward (execute : Call → CellHeap α → Option (CellHeap α)) (r : Request α) (m : Metadata)
    (scratch : CellHeap α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : ∀ h,Preparation h .forward live) (launch : Gate α)
    (viewX viewW : Except Failure Unit := .ok ()) : Outcome β (Context α × Tensor α) :=
  if (pythonChecks r m).all id then
    Host.forward execute r scratch live encode entry prepare launch viewX viewW
  else early entry .internal

/-- Saved-tensor/version retrieval occurs first, as in Python. Its decoded
Context contains current saved values. The forward adjoint additionally needs
those values to be unchanged from the successful forward call. -/
def backward (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (m : TensorMetadata) (live : Caller)
    (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : Except Failure (BackwardPrepared ctx live)) (w x z : Gate α)
    (copyG : Except Failure Unit := .ok ()) (savedRead : Except Failure Unit := .ok ()) :
    Outcome β (Tensor α × Tensor α) :=
  match savedRead with
  | .error e => early entry e
  | .ok _ => if (backwardChecks ctx g m).all id then
      Host.backward weight input zero ctx g live encode entry prepare w x z copyG
    else early entry .internal

theorem forward_admitted (execute : Call → CellHeap α → Option (CellHeap α)) (r : Request α) (m : Metadata)
    (ok : Admitted r m) (scratch : CellHeap α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : ∀ h,Preparation h .forward live) (launch : Gate α) :
    forward execute r m scratch live encode entry prepare launch=
      Host.forward execute r scratch live encode entry prepare launch := by
  simp only [Admitted] at ok
  simp [forward,ok]

theorem backward_admitted (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (m : TensorMetadata) (ok : BackwardAdmitted ctx g m)
    (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : Except Failure (BackwardPrepared ctx live)) (w x z : Gate α) :
    backward weight input zero ctx g m live encode entry prepare w x z=
      Host.backward weight input zero ctx g live encode entry prepare w x z := by
  simp only [BackwardAdmitted] at ok
  simp [backward,ok]

theorem forward_frame (execute : Call → CellHeap α → Option (CellHeap α)) (r : Request α) (m : Metadata)
    (scratch : CellHeap α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : ∀ h,Preparation h .forward live) (launch : Gate α)
    (viewX viewW : Except Failure Unit) (addr : Nat) (active : live addr) :
    (forward execute r m scratch live encode entry prepare launch viewX viewW).bytes addr=entry addr := by
  unfold forward
  split
  · exact Host.forward_frame _ _ _ _ _ _ _ _ _ _ addr active
  · rfl

theorem backward_frame (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (m : TensorMetadata) (live : Caller)
    (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : Except Failure (BackwardPrepared ctx live)) (w x z : Gate α)
    (copyG savedRead : Except Failure Unit) (addr : Nat) (active : live addr) :
    (backward weight input zero ctx g m live encode entry prepare w x z copyG savedRead).bytes addr=entry addr := by
  unfold backward
  split
  · rfl
  · split
    · exact Host.backward_frame _ _ _ _ _ _ _ _ _ _ _ _ _ addr active
    · rfl

/-- The CUDA x-grid limit discharges the historical native PID premise. -/
theorem checked_row_words (r : Request α) (m : Metadata) (ok : Admitted r m)
    (a : Native.Allocation (checkedCall r m ok).cells)
    (row : Fin (checkedCall r m ok).rows) (lane : Fin (checkedCall r m ok).block) :
    ((BitVec.ofInt 64 (BitVec.ofNat 32 row.val).toInt)*BitVec.ofNat 64 (checkedCall r m ok).cols+
      BitVec.ofNat 64 lane.val).toInt=(row.val*(checkedCall r m ok).cols+lane.val:Int) := by
  apply NativeGrid.row_words _ a row lane
  have lim:=checked_grid r m ok
  have small:=row.isLt
  omega

#axiomsClean forward_frame
#axiomsClean backward_frame
#axiomsClean checked_row_words
#axiomsClean saved_input_identity
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.CheckedHost

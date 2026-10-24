import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostExecution

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.HostStorage
open VeriTile.Triton Storage Host
noncomputable section
variable {α β : Type}

def mutate (live : Caller) (before changes : Heap β) : Heap β := by
  classical
  exact fun addr => if live addr then changes addr else before addr

/-- Saved inputs are direct views. Intervening writes are allowed only outside
all saved logical bytes; PyTorch version validation is a separate error gate.
Unlike the former snapshots, caller writes to X/W are not protected here. -/
def PreservesSaved (ctx : Context α) (mutations : Caller) : Prop :=
  ∀ role (saved : savedRole role) off,off<cells ctx.call role → ∀ byte,
    ¬mutations (address ctx.storage role (saved_forward role saved) off byte)

/-- Ordinary saved-tensor retrieval/decoding at the current backward entry.
This is an input representation relation, independent of program outputs,
caller visibility, and numerical laws. Invalid tensor versions are a separate
savedRead error before any copy/allocation/kernel. -/
def SavedAtEntry (ctx : Context α) (encode : α → Fin 4 → β) (entry : Heap β) : Prop :=
  ∀ role (saved : savedRole role) off,off<cells ctx.call role → ∀ byte,
    entry (address ctx.storage role (saved_forward role saved) off byte)=encode (ctx.saved (role,off)) byte

theorem saved_survives (ctx : Context α) (encode : α → Fin 4 → β) (entry changes : Heap β)
    (mutations : Caller) (current : SavedAtEntry ctx encode entry)
    (unchanged : PreservesSaved ctx mutations) :
    SavedAtEntry ctx encode (mutate mutations entry changes) := by
  intro role saved off inside byte
  rw [mutate,if_neg (unchanged role saved off inside byte)]
  exact current role saved off inside byte

theorem saved_backing_not_written (ctx : Context α) (live : Caller) (p : BackwardPrepared ctx live)
    (role : RegionName) (saved : savedRole role) (addr : Nat)
    (lo : (p.storage.allocation role (saved_backward role saved)).buffer.base≤addr)
    (hi : addr<(p.storage.allocation role (saved_backward role saved)).buffer.base+
      (p.storage.allocation role (saved_backward role saved)).buffer.extent*4) :
    ¬written p.storage addr := by
  rintro ⟨i,inside,location⟩
  have different : role≠i.1.val := by
    intro eq
    have output:=i.1.property
    rw [←eq] at output
    rcases saved with rfl|rfl|rfl <;> simp [outputs] at output
  have capacity:=inside.trans_le (p.storage.allocation i.1.val (output_member .backward _ i.1.property)).holds
  have bit:=i.2.2.isLt
  rcases p.storage.separate role (saved_backward role saved) i.1.val
      (output_member .backward _ i.1.property) different with sep|sep <;>
    unfold Storage.atIndex address at location <;> omega

/-- Backward never re-encodes saved buffers: every raw backing byte is retained,
including padding and caller-visible aliases, for any source values/result. -/
theorem backward_saved_bytes {ρ : Type} (ctx : Context α) (live : Caller)
    (p : BackwardPrepared ctx live) (encode : α → Fin 4 → β) (entry : Heap β)
    (actual : CellHeap α) (result : Except Failure ρ)
    (role : RegionName) (saved : savedRole role) (addr : Nat)
    (lo : (p.storage.allocation role (saved_backward role saved)).buffer.base≤addr)
    (hi : addr<(p.storage.allocation role (saved_backward role saved)).buffer.base+
      (p.storage.allocation role (saved_backward role saved)).buffer.extent*4) :
    (finish p.toPrepared encode entry actual result).bytes addr=entry addr :=
  Storage.commit_frame ctx.call .backward p.storage encode entry (fun r o => actual (r,o)) addr
    (saved_backing_not_written ctx live p role saved addr lo hi)

/-- Full wrapper outcomes preserve every byte in every reused saved backing,
including version errors, preparation-independent early errors, zero-fill
failures and partial weight/input launch failures. Visibility is unrestricted. -/
theorem backward_saved_full_frame (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (live : Caller) (p : BackwardPrepared ctx live)
    (encode : α → Fin 4 → β) (entry : Heap β) (w x z : Gate α)
    (resolve savedRead : Except Failure Unit) (role : RegionName) (saved : savedRole role) (addr : Nat)
    (lo : (ctx.storage.allocation role (saved_forward role saved)).buffer.base≤addr)
    (hi : addr<(ctx.storage.allocation role (saved_forward role saved)).buffer.base+
      (ctx.storage.allocation role (saved_forward role saved)).buffer.extent*4) :
    (Host.backward weight input zero ctx g live encode entry (.ok p) w x z resolve savedRead).bytes addr=entry addr := by
  have lower : (p.storage.allocation role (saved_backward role saved)).buffer.base≤addr := by
    rw [p.reuse role saved]; exact lo
  have upper : addr<(p.storage.allocation role (saved_backward role saved)).buffer.base+
      (p.storage.allocation role (saved_backward role saved)).buffer.extent*4 := by
    rw [p.reuse role saved]; exact hi
  unfold Host.backward
  repeat' (first | exact backward_saved_bytes ctx live p encode entry _ _ role saved addr lower upper | rfl | ((try dsimp only); split))

theorem backward_current_saved_readback {ρ : Type} (ctx : Context α) (live : Caller)
    (p : BackwardPrepared ctx live) (encode : α → Fin 4 → β) (entry : Heap β)
    (current : SavedAtEntry ctx encode entry) (actual : CellHeap α) (result : Except Failure ρ)
    (role : RegionName) (saved : savedRole role) (off : Nat) (inside : off<cells ctx.call role) (byte : Fin 4) :
    (finish p.toPrepared encode entry actual result).bytes
      (address p.storage role (saved_backward role saved) off byte)=encode (ctx.saved (role,off)) byte := by
  have capacity:=inside.trans_le (p.storage.allocation role (saved_backward role saved)).holds
  have bit:=byte.isLt
  rw [backward_saved_bytes ctx live p encode entry actual result role saved _
    (by unfold address; omega) (by unfold address; omega)]
  rw [saved_addresses ctx live p role saved off byte]
  exact current role saved off inside byte

theorem forward_readback (r : Request α) (h : Call) (live : Caller) (p : Prepared h .forward live)
    (actual : CellHeap α) (encode : α → Fin 4 → β) (entry : Heap β)
    (off : Nat) (inside : off<h.cells) (byte : Fin 4) :
    (finish p encode entry actual (.ok (makeContext r h p actual,outputTensor r h p actual))).bytes
      (address p.storage "y" (by simp [buffers]) off byte)=
      encode ((outputTensor r h p actual).values off) byte := by
  exact Storage.commit_output h .forward p.storage encode entry (fun r o => actual (r,o))
    "y" (by simp [outputs]) off (by simpa [cells] using inside) byte

theorem backward_readback (ctx : Context α) (live : Caller) (p : BackwardPrepared ctx live)
    (actual : CellHeap α) (encode : α → Fin 4 → β) (entry : Heap β) :
    (∀ off,off<ctx.call.cells → ∀ byte,
      (finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual))).bytes
        (address p.storage "dx" (by simp [buffers]) off byte)=
        encode ((gradientTensors ctx p actual).1.values off) byte) ∧
    (∀ off,off<ctx.call.cols → ∀ byte,
      (finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual))).bytes
        (address p.storage "dw" (by simp [buffers]) off byte)=
        encode ((gradientTensors ctx p actual).2.values off) byte) := by
  constructor
  · intro off inside byte
    exact Storage.commit_output ctx.call .backward p.storage encode entry (fun r o => actual (r,o))
      "dx" (by simp [outputs]) off (by simpa [cells] using inside) byte
  · intro off inside byte
    exact Storage.commit_output ctx.call .backward p.storage encode entry (fun r o => actual (r,o))
      "dw" (by simp [outputs]) off (by simpa [cells] using inside) byte

theorem returned_shapes (r : Request α) (h : Call) (live : Caller) (p : Prepared h .forward live)
    (actual : CellHeap α) (backwardLive : Caller) (bp : BackwardPrepared (makeContext r h p actual) backwardLive)
    (backwardValues : CellHeap α) :
    (outputTensor r h p actual).shape=r.shape.dims ∧
    (gradientTensors (makeContext r h p actual) bp backwardValues).1.shape=r.shape.dims ∧
    (gradientTensors (makeContext r h p actual) bp backwardValues).2.shape=r.weightShape ∧
    (outputTensor r h p actual).device=r.inputDevice ∧
    (gradientTensors (makeContext r h p actual) bp backwardValues).1.device=r.inputDevice ∧
    (gradientTensors (makeContext r h p actual) bp backwardValues).2.device=r.inputDevice := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

#axiomsClean saved_survives
#axiomsClean backward_saved_bytes
#axiomsClean backward_saved_full_frame
#axiomsClean backward_current_saved_readback
#axiomsClean saved_backing_not_written
#axiomsClean forward_readback
#axiomsClean backward_readback
#axiomsClean returned_shapes
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.HostStorage

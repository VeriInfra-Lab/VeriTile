import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Host
namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
open Physical (rowValues)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 6000000

/-- Ordinary Torch allocator alignment. This states no kernel value. -/
structure AllocationContract {M : Type} (rt : Runtime M) : Prop where
  zeros : ∀ rows cols owner before output after,
    rt.perform (.zeros rows cols owner) before = .ok output after →
    output.view.storage.baseByte % 4 = 0

theorem allocated_alignment {M : Type} (rt : Runtime M) (law : AllocationContract rt)
    (rows cols owner : Nat) (before after : World M) (output : Matrix)
    (allocated : rt.perform (.zeros rows cols owner) before = .ok output after)
    (r c : Nat) : (output.key r c).address % 4 = 0 := by
  have h := law.zeros rows cols owner before output after allocated
  simp only [Matrix.key,Allocation.key]
  omega

/-- The forward store graph equals a complete run of the full repaired row
source against evolving memory, using the original two checked read tensors. -/
theorem forward_complete {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (adapter : Storage.Adapter α κ) (a b : Tensor)
    (call : ForwardCall) (admit : ForwardContract a b)
    (ya : MatrixBinding a call.prediction) (ta : MatrixBinding b call.target)
    (output : Matrix) (rows : output.rows = call.prediction.rows)
    (cols : output.cols = outputCols call) (unit : output.colStride = 1)
    (backed : output.view.backed)
    (ypSeparate : call.prediction.view.storage.separate output.view.storage)
    (tpSeparate : call.target.view.storage.separate output.view.storage) (initial : M) :
    let p : Physical.ForwardLayout := ⟨call.prediction,call.target,output,
      blockSize call.prediction.cols,call.flag,modeCode call.mode⟩
    Grid.runRows ops (fun rc => output.key rc.1 rc.2)
      (fun s r c => Storage.forwardValue ops adapter s p (block_positive _) r c)
      output.cols (List.range output.rows) initial =
      Storage.forwardKernel ops adapter initial p (block_positive _) := by
  let p : Physical.ForwardLayout := ⟨call.prediction,call.target,output,
    blockSize call.prediction.cols,call.flag,modeCode call.mode⟩
  obtain ⟨targetRows,targetCols⟩ := ya.rows_cols ta admit.shape
  have valid : Grid.ForwardValid p :=
    ⟨(ya.contiguous admit.left.contiguous).1,(ta.contiguous admit.right.contiguous).1,
      unit,targetRows,targetCols,rows,by
        change output.cols = if modeCode call.mode = 0 then _ else _
        rw [cols]
        cases mode : call.mode <;> simp [outputCols,modeCode,p,mode],
      call.prediction.from_backed ya.backed,call.target.from_backed ta.backed,
      output.from_backed backed,ypSeparate,tpSeparate⟩
  have h := Grid.forward_rows ops adapter initial p (block_positive _) valid (List.range output.rows)
    (fun r hr => List.mem_range.mp hr)
  simpa [Storage.forwardKernel,Physical.jobs,List.product,Grid.scheduledJobs,Grid.rowJobs] using h

/-- Source row executions read the evolving physical state. Checked storage
separation proves their equivalence to the complete source-derived store graph;
no snapshot or output callback is required, including empty matrices. -/
theorem backward_rows {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (adapter : Storage.Adapter α κ) (a b g : Tensor)
    (call : BackwardCall) (admit : BackwardContract a b g)
    (ia : MatrixBinding a call.input) (ta : MatrixBinding b call.target)
    (initial : M) (schedule : List Nat) (active : ∀ r ∈ schedule, r < call.input.rows) :
    let p : Physical.BackwardLayout := ⟨call.target,call.input,blockSize call.input.cols,call.flag⟩
    Grid.runRows ops (fun rc => call.input.key rc.1 rc.2)
      (fun s r c => Storage.backwardValue ops adapter s p (block_positive _) r c)
      call.input.cols schedule initial =
    stores ops (fun rc => call.input.key rc.1 rc.2)
      (fun rc => Storage.backwardValue ops adapter initial p (block_positive _) rc.1 rc.2)
      (Grid.scheduledJobs call.input.cols schedule) initial := by
  let p : Physical.BackwardLayout := ⟨call.target,call.input,blockSize call.input.cols,call.flag⟩
  obtain ⟨rows,cols⟩ := ia.rows_cols ta admit.shape
  have targetUnit := (ta.contiguous admit.right.contiguous).1
  have stable : ∀ (done : List (Nat×Nat)),
      (∀ rc ∈ done, rc.1 < call.input.rows ∧ rc.2 < call.input.cols) →
      ∀ r c, r < call.input.rows → c < call.input.cols →
      Storage.backwardValue ops adapter
        (stores ops (fun rc => call.input.key rc.1 rc.2)
          (fun rc => Storage.backwardValue ops adapter initial p (block_positive _) rc.1 rc.2) done initial)
        p (block_positive _) r c = Storage.backwardValue ops adapter initial p (block_positive _) r c := by
    intro done doneValid r c hr hc
    have nonempty : a.metadata.numel ≠ 0 := by
      rw [←ia.toViewBinding.numel,call.input.numel]
      exact Nat.ne_of_gt (Nat.mul_pos (by omega) (by omega))
    have separate := separate_symm (ia.toViewBinding.separate ta.toViewBinding (admit.separate nonempty).1)
    rw [Storage.backward_formula ops adapter _ p (block_positive _) targetUnit r c hc
      (by rw [rows];exact hr) (by rw [cols];exact hc),
      Storage.backward_formula ops adapter initial p (block_positive _) targetUnit r c hc
      (by rw [rows];exact hr) (by rw [cols];exact hc)]
    rw [Grid.input_after_stores ops initial call.target call.input
      (call.target.from_backed ta.backed) (call.input.from_backed ia.backed) separate _ done doneValid r c
      (by rw [rows];exact hr) (by rw [cols];exact hc)]
  simpa only [stores,List.nil_append] using Grid.runRows_eq_stores ops (fun rc => call.input.key rc.1 rc.2)
    (fun s r c => Storage.backwardValue ops adapter s p (block_positive _) r c)
    call.input.rows call.input.cols initial stable schedule active [] (by simp)

/-- The modeled direct backward graph is the actual complete row-source run. -/
theorem backward_complete {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (adapter : Storage.Adapter α κ) (a b g : Tensor)
    (call : BackwardCall) (admit : BackwardContract a b g)
    (ia : MatrixBinding a call.input) (ta : MatrixBinding b call.target) (initial : M) :
    let p : Physical.BackwardLayout := ⟨call.target,call.input,blockSize call.input.cols,call.flag⟩
    Grid.runRows ops (fun rc => call.input.key rc.1 rc.2)
      (fun s r c => Storage.backwardValue ops adapter s p (block_positive _) r c)
      call.input.cols (List.range call.input.rows) initial =
      Storage.backwardKernel ops adapter initial p (block_positive _) := by
  have h := backward_rows ops adapter a b g call admit ia ta initial (List.range call.input.rows)
    (fun r hr => List.mem_range.mp hr)
  simpa [Storage.backwardKernel,Physical.jobs,List.product,Grid.scheduledJobs,Grid.rowJobs] using h

/-- The source's literal unit-column pointer is exact on active coordinates. -/
theorem source_pointer {t : Tensor} {m : Matrix} (binding : MatrixBinding t m)
    (admit : TensorContract t) (grid : L2NormBwd.Integer.GridCompatible m.rows)
    (r c : Nat) (hr : r < m.rows) (hc : c < m.cols) :
    (t.metadata.dataPtr + BitVec.ofNat 64 (r*m.rowStride)*BitVec.ofNat 64 4 +
      BitVec.ofNat 64 c*BitVec.ofNat 64 4).toNat = (m.key r c).address ∧
    (m.key r c).address % 4 = 0 ∧
    ∀ byte, byte < 4 → (m.key r c).address+byte < m.view.storage.baseByte+m.view.storage.bytes ∧
      (m.key r c).address+byte < 2^64 := by
  have nonempty : t.metadata.numel ≠ 0 := by
    rw [←binding.toViewBinding.numel,m.numel]
    exact Nat.ne_of_gt (Nat.mul_pos (by omega) (by omega))
  have ptr : t.metadata.dataPtr = BitVec.ofNat 64 (m.view.storage.baseByte+4*m.view.storageOffset) := by
    apply BitVec.eq_of_toNat_eq
    rw [←binding.pointer nonempty,BitVec.toNat_ofNat,Nat.mod_eq_of_lt t.metadata.dataPtr.isLt]
  have exactPointer := Index.active_pointer m binding.metadata (m.from_backed binding.backed) grid r c hr hc
  have unit := (binding.contiguous admit.contiguous).1.active r c hr hc
  have same : t.metadata.dataPtr + BitVec.ofNat 64 (r*m.rowStride)*BitVec.ofNat 64 4 +
      BitVec.ofNat 64 c*BitVec.ofNat 64 4 = L2NormBwd.Integer.pointer m r c := by
    rw [ptr]
    simp only [L2NormBwd.Integer.pointer,L2NormBwd.Integer.rowProduct,←BitVec.ofNat_mul,unit]
  exact ⟨by rw [same];exact exactPointer.1,binding.aligned admit r c hr hc,
    by simpa only [exactPointer.1] using exactPointer.2⟩

/-- The successful native setup supplies only its ordinary PID/metadata bounds.
All source loop masks and pointer bounds are then derived from checked inputs. -/
theorem checked_indices {t : Tensor} {m : Matrix} (binding : MatrixBinding t m)
    (admit : TensorContract t) (grid : L2NormBwd.Integer.GridCompatible m.rows)
    (r c k lane : Nat) (hr : r < m.rows) (hc : c < m.cols)
    (hk : k < VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds m.cols (blockSize m.cols))
    (hl : lane < blockSize m.cols) :
    (BitVec.ofNat 64 (k*blockSize m.cols)+BitVec.ofNat 64 lane).slt (BitVec.ofNat 64 m.cols) =
      decide (k*blockSize m.cols+lane < m.cols) ∧
    (BitVec.ofNat 64 (k*blockSize m.cols)+BitVec.ofNat 64 (blockSize m.cols)).toInt =
      (((k+1)*blockSize m.cols : Nat) : Int) := by
  have h := Launch.prepared_indices m binding.metadata (m.from_backed binding.backed)
    (binding.contiguous admit.contiguous).1 grid r c k lane hr hc hk hl
  exact h.2.2

#axiomsClean allocated_alignment
#axiomsClean forward_complete
#axiomsClean backward_rows
#axiomsClean backward_complete
#axiomsClean source_pointer
#axiomsClean checked_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Contiguous

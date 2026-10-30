import bench.tritonbench_g.triton_attention.Proofs.NativeExecution
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
open Memory
noncomputable section
set_option maxHeartbeats 4000000

@[simp] theorem load_writeTensor (c : Shape) (s : Memory α) (b : Buffer) (x : Coord c → α) :
    load c (writeTensor c s b x) b = fun h i e => x (coord c h i e) := by
  funext h i e
  exact writeTensor_readback _ _ _ _ _
@[simp] theorem load_writeTensor_other (c : Shape) (s : Memory α) (b : Buffer) (x : Coord c → α)
    (r : Buffer) (hr : r ≠ b) : load c (writeTensor c s b x) r = load c s r := by
  funext h i e
  exact writeTensor_other _ _ _ _ _ _ hr
@[simp] theorem load_writeRows_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (r : Buffer) (hr : r ≠ b) :
    load c (writeRows c s b x) r = load c s r := by
  funext h i e
  exact writeRows_other _ _ _ _ _ _ hr
@[simp] theorem loadRow_writeTensor_other (c : Shape) (s : Memory α) (b : Buffer) (x : Coord c → α)
    (r : Buffer) (hr : r ≠ b) : loadRow c (writeTensor c s b x) r = loadRow c s r := by
  funext h i
  exact writeTensor_other _ _ _ _ _ _ hr
@[simp] theorem loadRow_writeRows_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (r : Buffer) (hr : r ≠ b) :
    loadRow c (writeRows c s b x) r = loadRow c s r := by
  funext h i
  exact writeRows_other _ _ _ _ _ _ hr
@[simp] theorem loadRow_writeRows (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) :
    loadRow c (writeRows c s b x) b = fun h i => x (rowCoord c h i) := by
  funext h i
  exact writeRows_readback _ _ _ _ _

@[simp] theorem input_forward (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) (scale : α) :
    input c (forward c p owners s scale) = input c s := by
  simp [input,forward]

@[simp] theorem saved_forward (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) (scale : α) :
    saved c (forward c p owners s scale) = forwardReference c p owners (input c s) scale := by
  simp [saved,forward,forwardReference,coord,rowCoord]

/-- The entire backward input record is reconstructed from actual current
memory. Its only numerical dependencies are the pinned source-precision graph
from the original input arrays; no saved-value/result premise is supplied. -/
theorem backwardInput_connection (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) (scale : α) :
    backwardInput c (preprocess c p owners (forward c p owners s scale)) =
      wholeBackwardInput c p owners (input c s) scale := by
  unfold backwardInput preprocess wholeBackwardInput
  simp only [saved_forward]
  simp [forward,preprocessReference,forwardReference,input,coord,rowCoord]

@[simp] theorem backwardPrefix_other (c : Shape) (p : Ops α) (scale : α) (s : Memory α) (n : Nat)
    (r : Buffer) (o : Nat) (hq : r ≠ .dq) (hk : r ≠ .dk) (hv : r ≠ .dv) :
    backwardPrefix c p scale s n r o = s r o := by
  induction n with
  | zero => exact writeTensor_other _ _ _ _ _ _ hq
  | succ n ih => rw [backwardPrefix,backwardStep_other _ _ _ _ _ _ _ hq hk hv,ih]

/-- Every logical returned tensor element equals its independent seeded
native-instruction recurrence. This holds for arbitrary primitive semantics;
linking native instructions to a chosen bit profile is a separate local step. -/
theorem whole_values (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α)
    (scale : α) (i : Coord c) :
    run c p owners s scale .o (offset c i) =
      (forwardReference c p owners (input c s) scale).y i.1.val i.2.1.val i.2.2.val ∧
    run c p owners s scale .dq (offset c i) =
      dqReference c p (wholeBackwardInput c p owners (input c s) scale) scale
        c.blocks i.1.val i.2.1.val i.2.2.val ∧
    run c p owners s scale .dk (offset c i) =
      dkReference c p (wholeBackwardInput c p owners (input c s) scale) scale
        i.1.val i.2.1.val i.2.2.val ∧
    run c p owners s scale .dv (offset c i) =
      dvReference c p (wholeBackwardInput c p owners (input c s) scale) scale
        i.1.val i.2.1.val i.2.2.val := by
  have hi := (grid_coverage c i.2.1).1
  refine ⟨?_,?_,?_,?_⟩
  · simp [run,backwardPrefix_other,preprocess,forward,writeRows_other,writeTensor_other,
      writeTensor_readback]
  · rw [run,backwardPrefix_dq,backwardInput_connection]
  · rw [run,backwardPrefix_dk _ _ _ _ _ _ hi,backwardInput_connection]
  · rw [run,backwardPrefix_dv _ _ _ _ _ _ hi,backwardInput_connection]

#axiomsClean saved_forward
#axiomsClean backwardInput_connection
#axiomsClean whole_values
#specNonCircular wholeBackwardInput avoiding [forward,preprocess,backwardStep,backwardPrefix,run]
#specNonCircular dqReference avoiding [forward,preprocess,backwardStep,backwardPrefix,run]
#specNonCircular dkReference avoiding [forward,preprocess,backwardStep,backwardPrefix,run]
#specNonCircular dvReference avoiding [forward,preprocess,backwardStep,backwardPrefix,run]
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Native

import bench.repairs.tritonbench_g.kldiv_triton.Contract
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Execution

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Contiguous
open Host (blockSize modeCode block_positive)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds)
open SwigluTriton.Index (signed_ofNat)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 6000000

theorem row_raw {M α β : Type} (ops : Ops M α β) (neg : Negation α) (memory : M)
    (m : Matrix) (positive : m.view.negative=Bool.false) (r c : Nat) (hc : c<m.cols) :
    Physical.rowValues ops memory m r c=row ops neg memory m r c := by
  simp only [row,logical,Negation.encode,positive,Bool.false_eq_true,if_false,m.logical_key r c hc]
  rfl

theorem pid_exact (rows r : Nat) (dispatch : Dispatch rows) (hr : r<rows) :
    (BitVec.ofNat 32 r).toInt=(r:Int) :=
  signed_ofNat 32 r (by decide) (by unfold Dispatch L2NormBwd.Integer.GridCompatible at dispatch;omega)

theorem pid_extended (rows r : Nat) (dispatch : Dispatch rows) (hr : r<rows) :
    (BitVec.ofNat 32 r).signExtend 64=BitVec.ofNat 64 r := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),pid_exact rows r dispatch hr,
    signed_ofNat 64 r (by decide) (by unfold Dispatch L2NormBwd.Integer.GridCompatible at dispatch;omega)]

def sourcePointer (t : Tensor) (m : Matrix) (r c : Nat) : BitVec 64 :=
  t.metadata.dataPtr + ((BitVec.ofNat 32 r).signExtend 64 * BitVec.ofNat 64 m.rowStride)*BitVec.ofNat 64 4 +
    BitVec.ofNat 64 c*BitVec.ofNat 64 4

/-- Actual PID cast, signed row-stride multiplication, data pointer, active
alignment, four backed bytes and native64 bounds of every source access. -/
def Addresses (t : Tensor) (m : Matrix) : Prop :=
  ∀ r c,r<m.rows → c<m.cols →
    (sourcePointer t m r c).toNat=(m.key r c).address ∧
    (m.key r c).address%4=0 ∧
    (L2NormBwd.Integer.rowProduct r m.rowStride).toInt=(r*m.rowStride:Nat) ∧
    ∀ byte,byte<4 → (m.key r c).address+byte<m.view.storage.baseByte+m.view.storage.bytes ∧
      (m.key r c).address+byte<2^64

theorem addresses {t : Tensor} {m : Matrix} (binding : MatrixBinding t m)
    (admitted : TensorContract t) (dispatch : Dispatch m.rows) : Addresses t m := by
  intro r c hr hc
  have pointer := source_pointer binding admitted dispatch r c hr hc
  have exact := L2NormBwd.Integer.active_no_wrap m binding.metadata
    (m.from_backed binding.backed) dispatch r c hr hc
  refine ⟨?_,pointer.2.1,exact.2.2.2.1,pointer.2.2⟩
  simpa only [sourcePointer,pid_extended m.rows r dispatch hr,←BitVec.ofNat_mul] using pointer.1

/-- Every signed control value is exact, including masked lanes and the final
loop increment; the column bound follows from the actual four-byte backing. -/
def Control (m : Matrix) : Prop :=
  ∀ r,r<m.rows → (BitVec.ofNat 32 r).toInt=(r:Int) ∧
    ∀ k lane,k<rounds m.cols (blockSize m.cols) → lane<blockSize m.cols →
      (BitVec.ofNat 64 (k*blockSize m.cols)+BitVec.ofNat 64 lane).toInt=
        ((k*blockSize m.cols+lane:Nat):Int) ∧
      (BitVec.ofNat 64 (k*blockSize m.cols)+BitVec.ofNat 64 lane).slt (BitVec.ofNat 64 m.cols)=
        decide (k*blockSize m.cols+lane<m.cols) ∧
      (BitVec.ofNat 64 (k*blockSize m.cols)+BitVec.ofNat 64 (blockSize m.cols)).toInt=
        (((k+1)*blockSize m.cols:Nat):Int)

theorem control {t : Tensor} {m : Matrix} (binding : MatrixBinding t m)
    (admitted : TensorContract t) (dispatch : Dispatch m.rows) : Control m := by
  intro r hr
  refine ⟨pid_exact _ _ dispatch hr,?_⟩
  intro k lane hk hl
  have width := Index.columns_bound m (m.from_backed binding.backed)
    (binding.contiguous admitted.contiguous).1 (by omega)
  have bounds := Index.loop_bounds m.cols (blockSize m.cols) k lane
    (block_positive _) (Launch.block_bound _) width hk hl
  refine ⟨?_,Index.loop_mask_exact _ _ _ _ (block_positive _) (Launch.block_bound _) width hk hl,
    Index.loop_increment_exact _ _ _ (block_positive _) (Launch.block_bound _) width hk⟩
  rw [←BitVec.ofNat_add]
  exact signed_ofNat 64 _ (by decide) bounds.2.1

theorem ForwardDomain.valid {a b o : Tensor} {prediction target output : Matrix} {mode : Math.Mode}
    (d : ForwardDomain a b o prediction target output mode) (flag : Bool) :
    Grid.ForwardValid (forwardLayout prediction target output flag mode) := by
  obtain ⟨rows,cols⟩ := d.predictionBinding.rows_cols d.targetBinding d.admitted.shape
  refine ⟨(d.predictionBinding.contiguous d.admitted.left.contiguous).1,
    (d.targetBinding.contiguous d.admitted.right.contiguous).1,d.outputUnit,rows,cols,d.outputRows,?_,
    prediction.from_backed d.predictionBinding.backed,target.from_backed d.targetBinding.backed,
    output.from_backed d.outputBinding.backed,d.freshPrediction,d.freshTarget⟩
  change output.cols=if modeCode mode=0 then prediction.cols else rounds prediction.cols (blockSize prediction.cols)
  rw [d.outputCols]
  cases mode <;> rfl

#axiomsClean addresses
#axiomsClean control
#axiomsClean ForwardDomain.valid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel

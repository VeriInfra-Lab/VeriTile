import bench.repairs.tritonbench_g.chunk_gate_recurrence.IO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.PublicIO
import VeriTile.Triton.Memory.KernelSpec.TileIndex
import Mathlib.Data.Fintype.BigOperators

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.LogicalTensors
open VeriTile.Triton FP.Structural KernelProgram
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open scoped BigOperators
noncomputable section
set_option maxRecDepth 8192

/-- Both directions of the complete original source execution, safety,
observations and frame judgment, with actual tensor coordinate transport. -/
theorem full_canonical_iff (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) :
    (io kind c s seed p ⊨ fun a => outputs kind c s seed p
      (Reference.real kind c s seed p ((inputs kind c s seed p).symm a))) ↔
      (IO.program KernelRealReference.algebra kind c s seed p).Implements (Reference.real kind c s seed p) := by
  change (IO.program KernelRealReference.algebra kind c s seed p).Implements
    (fun raw => (outputs kind c s seed p).symm (outputs kind c s seed p
      (Reference.real kind c s seed p ((inputs kind c s seed p).symm (inputs kind c s seed p raw))))) ↔ _
  simp only [Equiv.symm_apply_apply]


theorem correct (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) (nonempty : 0<c.N) :
    io kind c s seed p ⊨ fun a => outputs kind c s seed p
      (Reference.real kind c s seed p ((inputs kind c s seed p).symm a)) :=
  (full_canonical_iff kind c s seed p).mpr (PublicIO.real kind c s seed p nonempty)

private theorem lane_length : (TileShape.allIndices [64,16]).length = 1024 := by
  rw [TileShape.length_allIndices]
  rfl

private theorem index_get (i : Fin 1024) :
    KernelRealReference.index i =
      (TileShape.allIndices [64,16]).get (Fin.cast lane_length.symm i) := by
  simp only [KernelRealReference.index, TileShape.reshapeIndex, TileShape.linearIndex,
    TileShape.product, Nat.mul_one, Nat.add_zero, List.get_eq_getElem, Fin.val_cast]

def indexEquiv : Fin 1024 ≃ Lane where
  toFun := KernelRealReference.index
  invFun := fun lane => Fin.cast lane_length (tilePos [64,16] lane)
  left_inv := by
    intro i
    dsimp only
    rw [index_get, tilePos_get]
    apply Fin.ext
    rfl
  right_inv := by
    intro lane
    rw [index_get]
    convert get_tilePos [64,16] lane using 1

theorem sum_index (f : Lane → ℝ) :
    (∑ k : Fin 1024, f (KernelRealReference.index k)) =
      ∑ key : Fin 64, ∑ value : Fin 16, f (key,value,PUnit.unit) := by
  calc
    _ = ∑ lane : Lane, f lane := indexEquiv.sum_comp f
    _ = _ := by simp [Fintype.sum_prod_type]

private def backwardResult (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (a : BackwardInputs c) : BackwardOutputs c where
  state := fun time key value => KernelRealReference.Formula.reverse
    (KernelRealReference.inputs (backwardArrays c s seed p a)) c.N (time.val+1) key value
  gate := fun time => ∑ key : Fin 64, ∑ value : Fin 16,
    KernelRealReference.Formula.reverse (KernelRealReference.inputs (backwardArrays c s seed p a))
      c.N (time.val+1) key value * a.saved time key value
  initial := fun key value => KernelRealReference.Formula.reverse
    (KernelRealReference.inputs (backwardArrays c s seed p a)) c.N 0 key value

/-- The full backward judgment after reindexing its complete DG contraction.
Only Real summation coordinates change; the FP reduction graph is untouched. -/
theorem backward_correct (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (nonempty : 0<c.N) :
    io .backward c s seed p ⊨ backwardResult c s seed p := by
  have target_eq : backwardResult c s seed p =
      (fun a => outputs .backward c s seed p
        (Reference.real .backward c s seed p ((inputs .backward c s seed p).symm a))) := by
    funext a
    change BackwardOutputs.mk _ _ _ = BackwardOutputs.mk _ _ _
    congr 1
    funext time
    change _ = ∑ k : Fin 1024,
      KernelRealReference.Formula.reverse (KernelRealReference.inputs (backwardArrays c s seed p a))
        c.N (time.val+1) (KernelRealReference.index k).1 (KernelRealReference.index k).2.1 *
          (backwardArrays c s seed p a).saved time.val (KernelRealReference.index k)
    rw [sum_index (fun lane => KernelRealReference.Formula.reverse
      (KernelRealReference.inputs (backwardArrays c s seed p a)) c.N (time.val+1)
        lane.1 lane.2.1 * (backwardArrays c s seed p a).saved time.val lane)]
    apply Finset.sum_congr rfl
    intro key _
    apply Finset.sum_congr rfl
    intro value _
    congr 1
    simp [backwardArrays, Reference.backwardInputs, IO.time, time.isLt,
      backwardInputs, Values.curryEquiv]
  exact (congrArg (LogicalProgramIO.Implements (io .backward c s seed p)) target_eq).mpr
    (correct .backward c s seed p nonempty)

#axiomsClean full_canonical_iff
#axiomsClean correct
#axiomsClean sum_index
#axiomsClean backward_correct
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.LogicalTensors

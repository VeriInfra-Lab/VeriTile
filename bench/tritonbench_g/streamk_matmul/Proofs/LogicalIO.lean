import bench.tritonbench_g.streamk_matmul.IO
import bench.tritonbench_g.streamk_matmul.Proofs.EventOriginalDomain
import bench.tritonbench_g.streamk_matmul.Proofs.EventCapacity
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.MatrixIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section

theorem cell_eq (p : Parameters) (i : C p) : EventIO.cell p i = finProdFinEquiv.symm i := by
  apply Prod.ext <;> apply Fin.ext <;> rfl

theorem a_index (p : Parameters) (r : Fin p.h.m) (k : Fin p.h.k) :
    finProdFinEquiv (r,k) = ⟨r.val*p.strideAM+k.val*p.strideAK,
      Geometry.input_mask_safe _ _ _ _ _ _ p.inputA r.isLt k.isLt⟩ := by
  apply Fin.ext
  change k.val+p.h.k*r.val = r.val*p.strideAM+k.val*p.strideAK
  simpa only [Nat.mul_comm,Nat.add_comm] using
    (Geometry.contiguous_offset _ _ _ _ _ _ p.inputA r.isLt k.isLt).symm

theorem b_index (p : Parameters) (k : Fin p.h.k) (c : Fin p.h.n) :
    finProdFinEquiv (k,c) = ⟨k.val*p.strideBK+c.val*p.strideBN,
      Geometry.input_mask_safe _ _ _ _ _ _ p.inputB k.isLt c.isLt⟩ := by
  apply Fin.ext
  change c.val+p.h.n*k.val = k.val*p.strideBK+c.val*p.strideBN
  simpa only [Nat.mul_comm,Nat.add_comm] using
    (Geometry.contiguous_offset _ _ _ _ _ _ p.inputB k.isLt c.isLt).symm

/-- Includes arbitrary initial C and scratch snapshots, complete source execution,
access safety, all output cells and frame, with no extra dimension assumptions. -/
theorem judgment_iff (p : Parameters) :
    (io p).Implements (fun xs row col => ∑ k : Fin p.h.k, xs.a row k * xs.b k col) ↔
    (EventIO.realIO p).Implements EventIO.realArithmetic (EventIO.realFormula p) := by
  have same : (io p).lift (fun xs row col => ∑ k : Fin p.h.k, xs.a row k * xs.b k col) =
      EventIO.realFormula p := by
    funext raw
    apply congrArg₂ (fun s w => (⟨s,w⟩ : (EventIO.realIO p).Outputs))
    · funext port
      fin_cases port
      funext i
      change (∑ k : Fin p.h.k,
        raw.scalars (0 : Fin 4) (finProdFinEquiv ((finProdFinEquiv.symm i).1,k)) *
        raw.scalars (1 : Fin 4) (finProdFinEquiv (k,(finProdFinEquiv.symm i).2))) = _
      simp only [EventIO.realFormula,LogicalIO.realFormula,Reference.matrixProduct,EventIO.outputIndex,cell_eq,EventIO.logicalInputs]
      apply Finset.sum_congr rfl
      intro k _
      rw [a_index,b_index]
      rfl
    · funext port
      exact Fin.elim0 port
  change (EventIO.realIO p).Implements EventIO.realArithmetic _ ↔ _
  rw [same]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.MatrixIO

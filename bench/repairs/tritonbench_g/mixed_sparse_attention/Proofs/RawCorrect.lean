import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelGrid
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.RealReturned

namespace VeriTile.Bench.Repairs.MixedSparseAttention.RawCorrect
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelGrid
noncomputable section

/-- Complete raw sparse-attention grid with independent sparse softmax values
and every memory cell outside its live output footprint. -/
theorem kernel_real (c : Config) (sep : Separated c) (layout : Contiguous c)
    (s : State Real.R) (values : RegionName → Nat → ℝ) (reads : Real.Reads c s values)
    (lengths : LengthsBound c s) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (order : work.Perm (programs (grid c))) :
    ∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) c (grid c) work s=some final ∧
      (∀ q : Raw.Coord c,live c s q → final.mem c.out (Raw.offset q)=.mk .real
        (some (Real.sparseExpected c s values (Raw.hz q) (Raw.row q) (Raw.dim q)))) ∧
      (∀ r i,¬Writes c s r i → final.mem r i=s.mem r i) := by
  obtain ⟨final,he,hr⟩:=KernelGrid.source (fun _=>Real.algebra) (fun _=>Real.dotModel)
    c sep layout s lengths hbm hbn work order
  refine ⟨final,he,?_,hr.frame⟩
  intro q hq
  have formula:=Real.source_value_real c layout s values reads hbm hbn
    (Raw.hz q) (Raw.row q) (Raw.dim q) q.2.2.isLt
  have active : Raw.row q<(s.mem (Region.cast c.seq) (Raw.hz q/c.H)).read .nat := hq
  simp only [if_pos active,Real.logicalExpected] at formula
  rw [hr.output q hq]
  exact congrArg (fun x : Real.R=>Cell.mk .real x) formula

#axiomsClean kernel_real
#specNonCircular Real.sparseExpected avoiding [Source.finalValues,Source.update,FP.Structural.exec,Whole.runPrograms,Whole.invoke]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.RawCorrect

import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeReads

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReads
open VeriTile Triton FP.Structural
open FPStorage (Memory state)
open FPStages (Stage diagonalLayout mergeLayout)
open NativeLaunch (start)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def DiagonalReads (c : FPDiagonal.Layout) (heads : Nat) (s : State α) (m : Memory α) : Prop :=
  ∀ r : RegionName,r="q" ∨ r="k" ∨ r="g" → ∀ off,off<heads*c.T*c.K →
    (s.mem r off).read .real=m r off

theorem diagonal_query (M : Algebra α) (split : Bool) (c : FPDiagonal.Layout)
    (heads : Nat) (s : State α) (m : Memory α) (hh : s.pids 2<heads)
    (reads : DiagonalReads c heads s m) (r : RegionName) (hr : r="q" ∨ r="g") :
    FPDiagonal.queryLoad M split c s r=
      FPDiagonal.queryLoad M split c (state m s.pids s.numPids) r := by
  funext i
  have rh : r≠"h" := by rcases hr with rfl|rfl <;> decide
  have port : r="q" ∨ r="k" ∨ r="g" := hr.elim Or.inl (Or.inr ∘ Or.inr)
  dsimp only [FPDiagonal.queryLoad,FPDiagonal.row,FPDiagonal.rowStart,
    FPDiagonal.chunk,FPDiagonal.block,FPDiagonal.feature,state]
  by_cases active : FPDiagonal.row split c s i.1.val<c.T ∧ FPDiagonal.feature split c s i.2.1.val<c.K
  · have hb := FPReadBounds.diagonal_query split c s heads i hh active.1 active.2
    dsimp only [FPDiagonal.row,FPDiagonal.rowStart,FPDiagonal.chunk,FPDiagonal.block,
      FPDiagonal.feature] at active hb
    rw [if_pos active,if_pos active]
    apply congrArg M.fp32Load
    exact (reads r port _ hb).trans (FPStorage.import_real _ _ _ rh).symm
  · dsimp only [FPDiagonal.row,FPDiagonal.rowStart,FPDiagonal.chunk,FPDiagonal.block,
      FPDiagonal.feature] at active
    rw [if_neg active,if_neg active]

theorem diagonal_key (M : Algebra α) (split : Bool) (c : FPDiagonal.Layout)
    (heads : Nat) (s : State α) (m : Memory α) (hh : s.pids 2<heads)
    (reads : DiagonalReads c heads s m) (r : RegionName) (hr : r="k" ∨ r="g")
    (j : Nat) (hj : j<FPDiagonal.columns split c s) (e : Nat) :
    FPDiagonal.keyLoad M split c s r j e=
      FPDiagonal.keyLoad M split c (state m s.pids s.numPids) r j e := by
  have rh : r≠"h" := by rcases hr with rfl|rfl <;> decide
  have port : r="q" ∨ r="k" ∨ r="g" := Or.inr hr
  dsimp only [FPDiagonal.keyLoad,FPDiagonal.keyAddress,FPDiagonal.rowStart,
    FPDiagonal.chunk,FPDiagonal.block,FPDiagonal.feature,state]
  by_cases active : FPDiagonal.feature split c s e<c.K
  · have hb := FPReadBounds.diagonal_key split c s heads j e hh hj active
    dsimp only [FPDiagonal.keyAddress,FPDiagonal.rowStart,FPDiagonal.chunk,FPDiagonal.block,
      FPDiagonal.feature] at active hb
    rw [if_pos active,if_pos active]
    apply congrArg (M.cast (some .fp32) .real .real)
    exact (reads r port _ hb).trans (FPStorage.import_real _ _ _ rh).symm
  · dsimp only [FPDiagonal.feature] at active
    rw [if_neg active,if_neg active]

theorem diagonal_value (M : Algebra α) (split : Bool) (c : FPDiagonal.Layout) (scale : ℝ)
    (heads : Nat) (s : State α) (m : Memory α) (hh : s.pids 2<heads)
    (reads : DiagonalReads c heads s m) (j : Nat) (hj : j<FPDiagonal.columns split c s) :
    FPDiagonal.value M split c scale s j=
      FPDiagonal.value M split c scale (state m s.pids s.numPids) j := by
  have product : FPDiagonal.product M split c s j=
      FPDiagonal.product M split c (state m s.pids s.numPids) j := by
    funext i
    simp only [FPDiagonal.product,diagonal_query M split c heads s m hh reads "q" (Or.inl rfl),
      diagonal_query M split c heads s m hh reads "g" (Or.inr rfl),
      diagonal_key M split c heads s m hh reads "k" (Or.inl rfl) j hj,
      diagonal_key M split c heads s m hh reads "g" (Or.inr rfl) j hj]
  funext i
  simp only [FPDiagonal.value,product]

theorem merge_value (M : Algebra α) (c : ProducerGeometry.Shape)
    (idx : GridIndex (DiagonalGeometry.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.merge c) s m) :
    FPMerge.value M (mergeLayout c) (start (.merge c) idx s)=
      FPMerge.value M (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) := by
  have loaded (f : Nat) (hf : f<Geometry.ceilDiv c.K c.BK) :
      FPMerge.loaded M (mergeLayout c) (start (.merge c) idx s) f=
        FPMerge.loaded M (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) f := by
    funext i
    dsimp only [FPMerge.loaded,FPMerge.row,FPMerge.inputAddress,start,state,Stage.grid,mergeLayout]
    by_cases active : FPMerge.row (mergeLayout c) (start (.merge c) idx s) i.1.val<c.T
    · have hb := FPReadBounds.merge_scratch c idx m f i hf active
      dsimp only [FPMerge.row,FPMerge.inputAddress,start,state,Stage.grid,mergeLayout] at active hb
      rw [if_pos active,if_pos active]
      exact (real_read reads "scratch" _ (by simp [NativeLaunch.inputExtents]) (by decide) _ hb).trans
        (FPStorage.import_real _ _ _ (by decide)).symm
    · dsimp only [FPMerge.row,start,Stage.grid,mergeLayout] at active
      rw [if_neg active,if_neg active]
  have accum : FPMerge.accumulated M (mergeLayout c) (start (.merge c) idx s)=
      FPMerge.accumulated M (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) := by
    apply fold_congr
    intro f _ hf a
    funext i
    simp only [FPMerge.update,loaded f hf]
  funext i
  simp only [FPMerge.value,accum]

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReads

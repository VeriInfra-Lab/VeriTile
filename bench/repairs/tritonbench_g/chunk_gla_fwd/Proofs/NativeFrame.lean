import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeLaunch

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeFrame
open VeriTile Triton FP.Structural
open FPStages (Stage diagonalLayout mergeLayout)
open NativeLaunch (start)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

/-- The actual arbitrary predecessor cells are preserved outside this stage's
finite write allocation. No import/export of the exterior is performed. -/
theorem program_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (s : State α) (r : RegionName) (off : Nat)
    (outside : r≠stage.region ∨ stage.size≤off) :
    (FPStages.program M D stage (start stage idx s)).mem r off=s.mem r off := by
  rcases outside with other|beyond
  · exact FPStages.program_frame M D stage (start stage idx s) r off other
  · cases stage with
    | inter c scale =>
      simp only [FPStages.program]
      split
      · apply FPInter.frame
        right
        intro i hi equal
        have hb : FPInter.outputAddress c (start (.inter c scale) idx s) i<c.heads*c.T*c.BT :=
          FPStages.inter_bound c idx (fun _ _=>(default : α)) i hi
        change c.heads*c.T*c.BT≤off at beyond
        omega
      · rfl
    | intra c BK scale =>
      simp only [FPStages.program]
      split
      · apply FPDiagonal.frame
        right
        intro j hj i hi equal
        have hb : FPDiagonal.outputAddress Bool.false (diagonalLayout c BK)
            (start (.intra c BK scale) idx s) j i<c.heads*c.T*c.BT :=
          FPStages.intra_bound c BK idx (fun _ _=>(default : α)) j i hj hi
        change c.heads*c.T*c.BT≤off at beyond
        omega
      · rfl
    | split c scale =>
      simp only [FPStages.program]
      split
      · apply FPDiagonal.frame
        right
        intro j hj i hi equal
        have hb : FPDiagonal.outputAddress Bool.true (diagonalLayout c c.BK)
            (start (.split c scale) idx s) j i<Geometry.ceilDiv c.K c.BK*c.heads*c.T*16 :=
          FPStages.split_bound c idx (fun _ _=>(default : α)) j i hj hi
        change Geometry.ceilDiv c.K c.BK*c.heads*c.T*16≤off at beyond
        omega
      · rfl
    | merge c =>
      simp only [FPStages.program]
      split
      · apply FPMerge.frame
        right
        intro i hi equal
        have hb : FPMerge.outputAddress (mergeLayout c) (start (.merge c) idx s) i<c.heads*c.T*c.BT :=
          FPStages.merge_bound c idx (fun _ _=>(default : α)) i hi
        change c.heads*c.T*c.BT≤off at beyond
        omega
      · rfl
    | output c scale =>
      apply FPOutput.frame
      right
      intro i hi equal
      have hb : FPOutput.outputAddress c (start (.output c scale) idx s) i<c.heads*c.T*c.V :=
        FPStages.output_bound c idx (fun _ _=>(default : α)) i hi
      change c.heads*c.T*c.V≤off at beyond
      omega

theorem source_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (s t : State α)
    (ran : (NativeLaunch.job M D stage idx).run s=some t)
    (r : RegionName) (off : Nat) (outside : r≠stage.region ∨ stage.size≤off) :
    t.mem r off=s.mem r off := by
  obtain ⟨u,run,mem⟩ := NativeLaunch.program_run M D stage idx s
  have same : t=u := Option.some.inj (ran.symm.trans run)
  subst u
  rw [mem]
  exact program_frame M D stage idx s r off outside

#axiomsClean source_frame
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeFrame

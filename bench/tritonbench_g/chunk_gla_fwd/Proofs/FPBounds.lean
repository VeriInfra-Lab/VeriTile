import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPStages
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPStages
open VeriTile Triton FP.Structural
open FPStorage (Memory state exportMemory)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

def Stage.size : Stage → Nat
  | .inter c _ | .intra c _ _ | .merge c => c.heads*c.T*c.BT
  | .split c _ => Geometry.ceilDiv c.K c.BK*c.heads*c.T*16
  | .output c _ => c.heads*c.T*c.V

theorem pid_bound (g : Grid) (idx : GridIndex g) (ax : Nat) (ha : ax<g.rank) :
    idx.toPids ax<g.dim ax := by
  simpa [GridIndex.toPids,ha] using (idx ⟨ax,ha⟩).isLt

theorem inter_bound (c : ProducerGeometry.Shape) (idx : GridIndex (ProducerGeometry.grid c))
    (m : Memory α) (i : FPInter.Index)
    (hi : FPInter.active c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) i) :
    FPInter.outputAddress c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) i<c.heads*c.T*c.BT := by
  have hh : idx.toPids 2<c.heads := by
    simpa [ProducerGeometry.grid,Grid.dim] using pid_bound (ProducerGeometry.grid c) idx 2 (by simp [ProducerGeometry.grid,Grid.rank])
  simpa only [FPInter.outputAddress,FPStorage.state,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
    using Geometry.attention_bound hh hi.1 hi.2

theorem intra_bound (c : ProducerGeometry.Shape) (BK : Nat) (idx : GridIndex (DiagonalGeometry.grid c))
    (m : Memory α) (j : Nat) (i : FPDiagonal.Index (diagonalLayout c BK))
    (hj : j<FPDiagonal.columns Bool.false (diagonalLayout c BK) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids))
    (hi : FPDiagonal.active Bool.false (diagonalLayout c BK) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) i) :
    FPDiagonal.outputAddress Bool.false (diagonalLayout c BK) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) j i<c.heads*c.T*c.BT := by
  have hh : idx.toPids 2<c.heads := by
    simpa [DiagonalGeometry.grid,Grid.dim] using pid_bound (DiagonalGeometry.grid c) idx 2 (by simp [DiagonalGeometry.grid,Grid.rank])
  have hc : idx.toPids 1<c.NC := by
    simpa [DiagonalGeometry.grid,Grid.dim] using pid_bound (DiagonalGeometry.grid c) idx 1 (by simp [DiagonalGeometry.grid,Grid.rank])
  have hj' : j<16 := lt_of_lt_of_le hj (Nat.min_le_left _ _)
  have hcol : idx.toPids 1*16+j<c.BT := by rw [c.bt_eq]; exact Geometry.packed_lt hc hj'
  have hb := Geometry.attention_bound hh hi hcol
  simpa only [FPDiagonal.outputAddress,FPDiagonal.outputBase,FPDiagonal.block,diagonalLayout,
    FPStorage.state,Bool.false_eq_true,if_false,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb

theorem split_bound (c : ProducerGeometry.Shape) (idx : GridIndex (SplitGeometry.grid c))
    (m : Memory α) (j : Nat) (i : FPDiagonal.Index (diagonalLayout c c.BK))
    (hj : j<FPDiagonal.columns Bool.true (diagonalLayout c c.BK) (state m idx.toPids (SplitGeometry.grid c).toNumPids))
    (hi : FPDiagonal.active Bool.true (diagonalLayout c c.BK) (state m idx.toPids (SplitGeometry.grid c).toNumPids) i) :
    FPDiagonal.outputAddress Bool.true (diagonalLayout c c.BK) (state m idx.toPids (SplitGeometry.grid c).toNumPids) j i<
      Geometry.ceilDiv c.K c.BK*c.heads*c.T*16 := by
  have hh : idx.toPids 2<c.heads := by
    simpa [SplitGeometry.grid,Grid.dim] using pid_bound (SplitGeometry.grid c) idx 2 (by simp [SplitGeometry.grid,Grid.rank])
  have hf : idx.toPids 0<Geometry.ceilDiv c.K c.BK := by
    simpa [SplitGeometry.grid,Grid.dim] using pid_bound (SplitGeometry.grid c) idx 0 (by simp [SplitGeometry.grid,Grid.rank])
  have hj' : j<16 := lt_of_lt_of_le hj (Nat.min_le_left _ _)
  have hb := Geometry.scratch_bound hf hh hi hj'
  simpa only [FPDiagonal.outputAddress,FPDiagonal.outputBase,diagonalLayout,FPStorage.state,
    if_true,Geometry.scratchAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,
    show (SplitGeometry.grid c).toNumPids 2=c.heads by simp [Grid.toNumPids,Grid.dim,Grid.rank,SplitGeometry.grid],Nat.add_comm]
    using hb

theorem merge_bound (c : ProducerGeometry.Shape) (idx : GridIndex (DiagonalGeometry.grid c))
    (m : Memory α) (i : FPMerge.Index (mergeLayout c))
    (hi : FPMerge.active (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) i) :
    FPMerge.outputAddress (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) i<c.heads*c.T*c.BT := by
  have hh : idx.toPids 2<c.heads := by
    simpa [DiagonalGeometry.grid,Grid.dim] using pid_bound (DiagonalGeometry.grid c) idx 2 (by simp [DiagonalGeometry.grid,Grid.rank])
  simpa only [FPMerge.outputAddress,mergeLayout,FPStorage.state,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
    using Geometry.attention_bound hh hi.1 hi.2

theorem output_bound (c : OutputGrid.Shape) (idx : GridIndex (OutputGrid.grid c))
    (m : Memory α) (i : FPOutput.Index c)
    (hi : FPOutput.active c (state m idx.toPids (OutputGrid.grid c).toNumPids) i) :
    FPOutput.outputAddress c (state m idx.toPids (OutputGrid.grid c).toNumPids) i<c.heads*c.T*c.V := by
  have hh : idx.toPids 2<c.heads := by
    simpa [OutputGrid.grid,Grid.dim] using pid_bound (OutputGrid.grid c) idx 2 (by simp [OutputGrid.grid,Grid.rank])
  simpa only [FPOutput.outputAddress,FPStorage.state,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
    using Geometry.attention_bound hh hi.1 hi.2

theorem program_bounded (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (m : Memory α) (r : RegionName) (off : Nat)
    (ho : stage.size ≤ off) :
    (program M D stage (state m idx.toPids stage.grid.toNumPids)).mem r off=
      (state m idx.toPids stage.grid.toNumPids).mem r off := by
  cases stage with
  | inter c scale =>
    simp only [program]
    split
    · apply FPInter.frame
      right
      intro i hi he
      have hb := inter_bound c idx m i hi
      exact Nat.not_lt_of_ge ho (he ▸ hb)
    · rfl
  | intra c BK scale =>
    simp only [program]
    split
    · apply FPDiagonal.frame
      right
      intro j hj i hi he
      have hb := intra_bound c BK idx m j i hj hi
      exact Nat.not_lt_of_ge ho (he ▸ hb)
    · rfl
  | split c scale =>
    simp only [program]
    split
    · apply FPDiagonal.frame
      right
      intro j hj i hi he
      have hb := split_bound c idx m j i hj hi
      exact Nat.not_lt_of_ge ho (he ▸ hb)
    · rfl
  | merge c =>
    simp only [program]
    split
    · apply FPMerge.frame
      right
      intro i hi he
      have hb := merge_bound c idx m i hi
      exact Nat.not_lt_of_ge ho (he ▸ hb)
    · rfl
  | output c scale =>
    apply FPOutput.frame
    right
    intro i hi he
    have hb := output_bound c idx m i hi
    exact Nat.not_lt_of_ge ho (he ▸ hb)

theorem referenceStep_bounded (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (m : Memory α) (r : RegionName) (off : Nat)
    (hn : r≠stage.region ∨ stage.size ≤ off) : referenceStep M D stage idx m r off=m r off := by
  rcases hn with hr|ho
  · exact referenceStep_frame M D stage idx m r off hr
  · unfold referenceStep exportMemory
    rw [program_bounded M D stage idx m r off ho]
    exact FPStorage.payload_cell _ _

theorem reference_bounded (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (m : Memory α) (r : RegionName) (off : Nat) (hn : r≠stage.region ∨ stage.size ≤ off) :
    reference M D stage m r off=m r off :=
  FPGrid.reference_frame _ _ _ _ _ (fun idx _ mm => referenceStep_bounded M D stage idx mm r off hn)

#axiomsClean reference_bounded
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPStages

import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPBounds
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPReadBounds
open VeriTile Triton FP.Structural
open FPStorage (Memory state)
open FPStages (diagonalLayout mergeLayout pid_bound)
noncomputable section
variable {α : Type} [Inhabited α]

/-- Shared packed bound for all masked q/k/g and value loads. -/
theorem tensor_bound {heads T K h t e : Nat} (hh : h<heads) (ht : t<T) (he : e<K) :
    h*(T*K)+t*K+e<heads*T*K := by
  simpa [Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc] using Geometry.attention_bound hh ht he

theorem inter_reference (c : ProducerGeometry.Shape) (idx : GridIndex (ProducerGeometry.grid c))
    (m : Memory α) (f e : Nat)
    (hen : FPInter.enabled c (state m idx.toPids (ProducerGeometry.grid c).toNumPids))
    (he : FPInter.feature c f e<c.K) :
    idx.toPids 2*(c.T*c.K)+FPInter.rowStart c (state m idx.toPids (ProducerGeometry.grid c).toNumPids)*c.K+
      FPInter.feature c f e<c.heads*c.T*c.K := by
  apply tensor_bound _ hen.1 he
  simpa [ProducerGeometry.grid,Grid.dim] using pid_bound (ProducerGeometry.grid c) idx 2 (by simp [ProducerGeometry.grid,Grid.rank])

theorem inter_query (c : ProducerGeometry.Shape) (idx : GridIndex (ProducerGeometry.grid c))
    (m : Memory α) (f : Nat) (i : TileIndex [16,c.BK])
    (hr : FPInter.row c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) i.1.val<c.T)
    (he : FPInter.feature c f i.2.1.val<c.K) :
    idx.toPids 2*(c.T*c.K)+FPInter.row c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) i.1.val*c.K+
      FPInter.feature c f i.2.1.val<c.heads*c.T*c.K := by
  apply tensor_bound _ hr he
  simpa [ProducerGeometry.grid,Grid.dim] using pid_bound (ProducerGeometry.grid c) idx 2 (by simp [ProducerGeometry.grid,Grid.rank])

theorem inter_key (c : ProducerGeometry.Shape) (idx : GridIndex (ProducerGeometry.grid c))
    (m : Memory α) (f : Nat) (i : TileIndex [c.BK,16])
    (he : FPInter.feature c f i.1.val<c.K)
    (hr : FPInter.columnStart c (state m idx.toPids (ProducerGeometry.grid c).toNumPids)+i.2.1.val<c.T) :
    idx.toPids 2*(c.T*c.K)+FPInter.feature c f i.1.val+
      (FPInter.columnStart c (state m idx.toPids (ProducerGeometry.grid c).toNumPids)+i.2.1.val)*c.K<c.heads*c.T*c.K := by
  have hh : idx.toPids 2<c.heads := by
    simpa [ProducerGeometry.grid,Grid.dim] using pid_bound (ProducerGeometry.grid c) idx 2 (by simp [ProducerGeometry.grid,Grid.rank])
  simpa only [Nat.add_right_comm] using tensor_bound hh hr he

theorem diagonal_column (split : Bool) (c : FPDiagonal.Layout) (s : State α) (j : Nat)
    (hj : j<FPDiagonal.columns split c s) : FPDiagonal.rowStart split c s+j<c.T := by
  have h := lt_of_lt_of_le hj (Nat.min_le_right _ _)
  unfold FPDiagonal.rowStart
  omega

theorem diagonal_key (split : Bool) (c : FPDiagonal.Layout) (s : State α) (heads j e : Nat)
    (hh : s.pids 2<heads) (hj : j<FPDiagonal.columns split c s) (he : FPDiagonal.feature split c s e<c.K) :
    FPDiagonal.keyAddress split c s j e<heads*c.T*c.K := by
  have h := tensor_bound hh (diagonal_column split c s j hj) he
  dsimp only [FPDiagonal.keyAddress]
  rw [Nat.add_mul] at h
  omega

theorem diagonal_query (split : Bool) (c : FPDiagonal.Layout) (s : State α) (heads : Nat)
    (i : TileIndex [c.BC,c.BK]) (hh : s.pids 2<heads)
    (hr : FPDiagonal.row split c s i.1.val<c.T) (he : FPDiagonal.feature split c s i.2.1.val<c.K) :
    s.pids 2*(c.T*c.K)+FPDiagonal.row split c s i.1.val*c.K+FPDiagonal.feature split c s i.2.1.val<heads*c.T*c.K :=
  tensor_bound hh hr he

theorem merge_scratch (c : ProducerGeometry.Shape) (idx : GridIndex (DiagonalGeometry.grid c))
    (m : Memory α) (f : Nat) (i : FPMerge.Index (mergeLayout c))
    (hf : f<Geometry.ceilDiv c.K c.BK)
    (hr : FPMerge.row (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) i.1.val<c.T) :
    FPMerge.inputAddress (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) f i<
      Geometry.ceilDiv c.K c.BK*c.heads*c.T*16 := by
  have hh : idx.toPids 2<c.heads := by
    simpa [DiagonalGeometry.grid,Grid.dim] using pid_bound (DiagonalGeometry.grid c) idx 2 (by simp [DiagonalGeometry.grid,Grid.rank])
  have hj : i.2.1.val<16 := i.2.1.isLt
  have hb := Geometry.scratch_bound hf hh hr hj
  simpa only [FPMerge.inputAddress,mergeLayout,FPStorage.state,Geometry.scratchAddress,Nat.add_mul,Nat.mul_assoc,
    show (DiagonalGeometry.grid c).toNumPids 2=c.heads by simp [Grid.toNumPids,Grid.dim,Grid.rank,DiagonalGeometry.grid],
    Nat.add_assoc,Nat.add_comm] using hb

theorem output_features (c : OutputGrid.Shape) (idx : GridIndex (OutputGrid.grid c))
    (m : Memory α) (f : Nat) (i : TileIndex [c.BT,c.BK])
    (hr : FPOutput.row c (state m idx.toPids (OutputGrid.grid c).toNumPids) i.1.val<c.T)
    (he : FPOutput.feature c f i.2.1.val<c.K) :
    idx.toPids 2*(c.T*c.K)+FPOutput.row c (state m idx.toPids (OutputGrid.grid c).toNumPids) i.1.val*c.K+
      FPOutput.feature c f i.2.1.val<c.heads*c.T*c.K := by
  apply tensor_bound _ hr he
  simpa [OutputGrid.grid,Grid.dim] using pid_bound (OutputGrid.grid c) idx 2 (by simp [OutputGrid.grid,Grid.rank])

theorem output_history (c : OutputGrid.Shape) (idx : GridIndex (OutputGrid.grid c))
    (m : Memory α) (f : Nat) (i : TileIndex [c.BK,c.BV])
    (he : FPOutput.feature c f i.1.val<c.K)
    (hv : FPOutput.column c (state m idx.toPids (OutputGrid.grid c).toNumPids) i.2.1.val<c.V) :
    idx.toPids 2*(Geometry.ceilDiv c.T c.BT*c.K*c.V)+idx.toPids 1*c.K*c.V+
      FPOutput.feature c f i.1.val*c.V+FPOutput.column c (state m idx.toPids (OutputGrid.grid c).toNumPids) i.2.1.val<
      c.heads*(Geometry.ceilDiv c.T c.BT*c.K)*c.V := by
  have hh : idx.toPids 2<c.heads := by
    simpa [OutputGrid.grid,Grid.dim] using pid_bound (OutputGrid.grid c) idx 2 (by simp [OutputGrid.grid,Grid.rank])
  have ht : idx.toPids 1<Geometry.ceilDiv c.T c.BT := by
    simpa [OutputGrid.grid,Grid.dim] using pid_bound (OutputGrid.grid c) idx 1 (by simp [OutputGrid.grid,Grid.rank])
  have hk := Geometry.packed_lt ht he
  simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using tensor_bound hh hk hv

theorem output_attention (c : OutputGrid.Shape) (idx : GridIndex (OutputGrid.grid c))
    (m : Memory α) (i : TileIndex [c.BT,c.BT])
    (hr : FPOutput.row c (state m idx.toPids (OutputGrid.grid c).toNumPids) i.1.val<c.T) :
    idx.toPids 2*c.T*c.BT+FPOutput.row c (state m idx.toPids (OutputGrid.grid c).toNumPids) i.1.val*c.BT+i.2.1.val<
      c.heads*c.T*c.BT := by
  have hh : idx.toPids 2<c.heads := by
    simpa [OutputGrid.grid,Grid.dim] using pid_bound (OutputGrid.grid c) idx 2 (by simp [OutputGrid.grid,Grid.rank])
  simpa only [Nat.mul_assoc] using tensor_bound hh hr i.2.1.isLt

#axiomsClean diagonal_key
#axiomsClean merge_scratch
#axiomsClean output_history
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPReadBounds

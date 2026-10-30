import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicPackedInputs
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicReadCongruence

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type}

def reference (a : Scalar α) (s : Shape) (tile : Tile)
    (xs : ChannelValues α (backwardInputs s)) : State (Expr α) :=
  initial s (packedBases s) (packedMemory a.ops.zero s xs) tile

def forwardReference (a : Scalar α) (s : Shape) (tile : Tile)
    (xs : ChannelValues α (forwardInputs s)) : State (Expr α) :=
  reference a s tile (extendForward a.ops.zero s xs)

theorem reference_binding (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (xs : ChannelValues α (backwardInputs s)) :
    (backwardIO a p s tile).InputBinding xs (reference a s tile xs) := by
  intro port j active
  fin_cases port <;> rcases j with ⟨bh,t,e,⟨⟩⟩
  · change Expr.input (packedMemory a.ops.zero s xs ((packedBases s).q+dense s.T s.DK bh.val t.val e.val))=_
    rw [packed_query a.ops.zero s xs bh.val t.val e.val bh.isLt t.isLt e.isLt]
    rfl
  · change Expr.input (packedMemory a.ops.zero s xs ((packedBases s).k+dense s.T s.DK bh.val t.val e.val))=_
    rw [packed_key a.ops.zero s xs bh.val t.val e.val bh.isLt t.isLt e.isLt]
    rfl
  · change Expr.input (packedMemory a.ops.zero s xs ((packedBases s).v+dense s.T s.DV bh.val t.val e.val))=_
    rw [packed_value a.ops.zero s xs bh.val t.val e.val bh.isLt t.isLt e.isLt]
    rfl
  · change Expr.input (packedMemory a.ops.zero s xs ((packedBases s).g+dense s.T s.DV bh.val t.val e.val))=_
    rw [packed_gradient a.ops.zero s xs bh.val t.val e.val bh.isLt t.isLt e.isLt]
    rfl

theorem forward_reference_binding (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (xs : ChannelValues α (forwardInputs s)) :
    (forwardIO a p s tile).InputBinding xs (forwardReference a s tile xs) := by
  have h := reference_binding a p s tile (extendForward a.ops.zero s xs)
  intro port j active
  fin_cases port
  · exact h (0 : Fin 4) j trivial
  · exact h (1 : Fin 4) j trivial
  · exact h (2 : Fin 4) j trivial

theorem dense_matrix {β : Type} [Inhabited β] (s t : State β) (r : RegionName)
    (bh time width : Nat)
    (same : ∀ i j, i<time → j<width →
      (s.mem r (dense time width bh i j)).read .real=
        (t.mem r (dense time width bh i j)).read .real) :
    PublicRead.MatrixEqual s t r (bh*(time*width)) time width width 1 := by
  intro i j hi hj
  simpa only [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using same i j hi hj

theorem transposed_matrix {β : Type} [Inhabited β] (s t : State β) (r : RegionName)
    (bh time width : Nat)
    (same : ∀ i j, i<time → j<width →
      (s.mem r (dense time width bh i j)).read .real=
        (t.mem r (dense time width bh i j)).read .real) :
    PublicRead.MatrixEqual s t r (bh*(time*width)) width time 1 width := by
  intro i j hi hj
  simpa [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
    using same j i hj hi

theorem forward_values_binding (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) (xs : ChannelValues α (forwardInputs s))
    (state : State (Expr α)) (binding : (forwardIO a p s tile).InputBinding xs state) :
    FPStages.forwardValues algebra dots (config s 128) ((forwardInvocation α s tile).initial state)=
      FPStages.forwardValues algebra dots (config s 128) (forwardReference a s tile xs) := by
  have refBinding := forward_reference_binding a p s tile xs
  apply PublicRead.forward_values algebra dots (config s 128) _ _ (by rfl)
  · apply dense_matrix
    intro i j hi hj
    exact (binding (0 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (0 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm
  · apply transposed_matrix
    intro i j hi hj
    exact (binding (1 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (1 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm
  · apply dense_matrix
    intro i j hi hj
    exact (binding (2 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (2 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm

theorem query_values_binding (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (xs : ChannelValues α (backwardInputs s))
    (state : State (Expr α)) (binding : (backwardIO a p s tile).InputBinding xs state) :
    FPQuery.values algebra dots (backwardConfig s tile) ((backwardInvocation α s tile).initial state)=
      FPQuery.values algebra dots (backwardConfig s tile) (reference a s tile xs) := by
  have refBinding := reference_binding a p s tile xs
  apply PublicRead.query_values algebra dots (backwardConfig s tile)
  · apply dense_matrix
    intro i j hi hj
    exact (binding (3 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (3 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm
  · apply dense_matrix
    intro i j hi hj
    exact (binding (1 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (1 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm
  · apply transposed_matrix
    intro i j hi hj
    exact (binding (2 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (2 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm

theorem backward_values_binding (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (xs : ChannelValues α (backwardInputs s))
    (state : State (Expr α)) (binding : (backwardIO a p s tile).InputBinding xs state) :
    FPStages.backwardValues algebra dots (backwardConfig s tile) ((backwardInvocation α s tile).initial state)=
      FPStages.backwardValues algebra dots (backwardConfig s tile) (reference a s tile xs) := by
  have refBinding := reference_binding a p s tile xs
  apply PublicRead.backward_values algebra dots (backwardConfig s tile)
  · apply transposed_matrix
    intro i j hi hj
    exact (binding (0 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (0 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm
  · apply transposed_matrix
    intro i j hi hj
    exact (binding (3 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (3 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm
  · apply dense_matrix
    intro i j hi hj
    exact (binding (1 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (1 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm
  · apply dense_matrix
    intro i j hi hj
    exact (binding (2 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).trans
      (refBinding (2 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial).symm

#axiomsClean reference_binding
#axiomsClean forward_values_binding
#axiomsClean query_values_binding
#axiomsClean backward_values_binding
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO

import bench.tritonbench_g.chunk_cumsum_vector.Sources.Mathematical_RealStep
import VeriTile.Triton.Memory.Frame.Basic

/-! The whole real time loop. Its invariant derives the column carry from
initial inputs and unspecified values, preserves earlier outputs and frames
complete memory cells outside the active output window. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealLoop
open VeriTile.Triton RealProgram RealStep
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def carry (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (j : Fin p.BS) : ℝ :=
  RealMath.carry (fun t => value p l a padding t j) p.BT c

def expected (p : Layout) (l : Launch) (a : Nat → ℝ) (t : Nat) (j : Fin p.BS) : ℝ :=
  RealMath.prefixValue (column p l a j) p.T t

/-- Only active cells must have distinct addresses, across the whole time span. -/
def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (t u : Nat) (i j : Fin p.BS), RealStep.active p l t i.val → RealStep.active p l u j.val →
    address p l t i.val = address p l u j.val → t = u ∧ i = j

def footprint (p : Layout) (l : Launch) (o : RegionName) : WriteFootprint :=
  fun cell => cell.1 = o ∧ ∃ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val ∧
    cell.2 = address p l t j.val

theorem next_carry (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) :
    nextCarry p l a padding c (carry p l a padding c) = carry p l a padding (c + 1) := by
  funext j
  exact (RealMath.carry_succ (fun t => value p l a padding t j) p.BT c).symm

/-- Only the observed feature must be active. Other columns retain arbitrary
padding, but a triangular dot does not mix different feature columns. -/
theorem prefix_at_chunk (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (i : Fin p.BT) (j : Fin p.BS) (hi : c * p.BT + i.val < p.T)
    (hj : l.feature * p.BS + j.val < p.S) :
    carry p l a padding c j + chunkPrefix p l a padding c i j =
      expected p l a (c * p.BT + i.val) j := by
  rw [add_comm]
  refine (RealMath.prefix_from_chunk (fun t => value p l a padding t j)
    p.T p.BT c i hi).trans ?_
  simp only [expected, RealMath.prefixValue]
  apply Finset.sum_congr rfl
  intro k hk
  have ht := (Finset.mem_filter.mp hk).1
  have ht' := Finset.mem_range.mp ht
  simp [RealStep.value, RealStep.active, ht', hj, RealStep.column]

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (c : Nat) (u : BlockState) : Prop where
  undef : u.undef = initial.undef
  count_le : c ≤ numChunks p
  feature : u.regs .nat [] "i_s" = some (Tile.scalar l.feature)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  triangular : u.regs .real [p.BT, p.BT] "m_s" = some (lowerTriTile p.BT)
  accumulated : u.regs .real [p.BS] "b_z" =
    some (Tile.vec fun j => some (carry p l a (initial.undef s) c j))
  input : ∀ off, u.readMem s off = a off
  output : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → t < c * p.BT →
    u.readMem o (address p l t j.val) = expected p l a t j
  frame : BlockState.WriteWithin (footprint p l o) initial u

/-- The scatter changes only active cells of the current chunk. -/
theorem scatter_frame (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (u : BlockState) :
    BlockState.WriteWithin (footprint p l o) u (scatter p l o a padding c acc u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl, c * p.BT + idx.1.val, idx.2.1, ha, rfl⟩

theorem stop_eval (p : Layout) (u : BlockState) :
    evalOp (stopOp p) u = some (Tile.scalar (numChunks p)) := by
  simp only [stopOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealLoop

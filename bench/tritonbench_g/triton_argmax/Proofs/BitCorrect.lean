import bench.tritonbench_g.triton_argmax.Proofs.BitBackend
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonArgmax.BitCorrect
open VeriTile.Triton FP.Structural TypedSource MatrixReduction.MachineMax
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

def channel : Storage → FloatDType | .fp16 => .fp16 | .fp32 => .real

def stage2Input (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (i : Fin B) : Bits storage :=
  if i.val < N then toFloat (channel storage)
    ((s.mem X i.val).read (channel storage).toTileDType)
  else negativeInfinity storage

def stage2Selected (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (hB : 0 < B) : Fin B :=
  best storage (stage2Input storage s X N B) hB

theorem stage2_run (storage : Storage) (rest : Algebra (Bits storage))
    (s : State (Bits storage)) (X : RegionName) (I O : Region .int)
    (N B : Nat) (hB : 0 < B) :
    ∃ t, FP.Structural.exec (BitBackend.backend storage rest) (stage2 storage X I O N B) s = some t ∧
      t.mem = (s.write (Region.cast O) 0 (.mk .int
        ((s.mem (Region.cast I) (stage2Selected storage s X N B hB).val).read .int))).mem := by
  cases storage <;>
    simp [stage2, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, BitBackend.backend,
      BitBackend.indexed,
      State.setReg, TileShape.axisDim, hB, TileShape.insertAxisIndex,
      TileShape.eraseAxis, stage2Selected, stage2Input, channel,
      ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, natLt,
      Nat.ne_of_gt hB, TileShape.allIndices]
  all_goals rfl

def stage1Input (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (i : Fin B) : Bits storage :=
  if s.pids 0 * B + i.val < N then toFloat (channel storage)
    ((s.mem X (s.pids 0 * B + i.val)).read (channel storage).toTileDType)
  else negativeInfinity storage

def stage1Selected (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (hB : 0 < B) : Fin B :=
  best storage (stage1Input storage s X N B) hB

def stage1Memory (storage : Storage) (s : State (Bits storage)) (X Y : RegionName)
    (I : Region .int) (N B : Nat) (hB : 0 < B) : RegionName → Nat → Cell (Bits storage) :=
  let j := stage1Selected storage s X N B hB
  ((s.write Y (s.pids 0) (.mk (channel storage).toTileDType
    (ofFloat (channel storage) (stage1Input storage s X N B j)))).write
    (Region.cast I) (s.pids 0) (.mk .int (Int.ofNat (j.val + s.pids 0 * B)))).mem

theorem stage1_run (storage : Storage) (rest : Algebra (Bits storage))
    (s : State (Bits storage)) (X Y : RegionName) (I : Region .int)
    (N B : Nat) (index64 : Bool) (hB : 0 < B) :
    ∃ t, FP.Structural.exec (BitBackend.backend storage rest)
        (stage1 storage X Y I N B index64) s = some t ∧
      t.mem = stage1Memory storage s X Y I N B hB := by
  cases storage <;> cases index64 <;>
    simp [stage1, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, BitBackend.backend,
      BitBackend.indexed,
      State.setReg, TileShape.axisDim, hB, TileShape.insertAxisIndex,
      TileShape.eraseAxis, stage1Selected, stage1Input, stage1Memory, channel,
      ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, numeric,
      numericLt, natLt, Nat.ne_of_gt hB, TileShape.allIndices]
  all_goals rfl

/-- Independent machine argmax: largest comparison key, with the smallest lane
index on numerical ties, including a tie between positive and negative zero. -/
def IsArgMaximum {B : Nat} (storage : Storage) (xs : Fin B → Bits storage) (j : Fin B) : Prop :=
  (∀ i, rank storage (xs i) ≤ rank storage (xs j)) ∧
  (∀ i, rank storage (xs i) = rank storage (xs j) → j.val ≤ i.val)

/-- Complete first-stage source, including both index-width branches. The
stored value is an independently maximal padded input and the signed output
index is its smallest local lane plus the program's input offset. The original
input bits, including either zero sign, are preserved at the selected lane. -/
theorem argmax_stage1_bit_correctness
    (storage : Storage) (rest : Algebra (Bits storage))
    (s : State (Bits storage)) (X Y : RegionName) (I : Region .int)
    (N B : Nat) (index64 : Bool) (hB : 0 < B) (hYI : Y ≠ Region.cast I)
    (hnotNaN : ∀ i, isNaN storage (stage1Input storage s X N B i) = Bool.false) :
    ∃ t, FP.Structural.exec (BitBackend.backend storage rest)
        (stage1 storage X Y I N B index64) s = some t ∧
      (∃ j : Fin B, IsArgMaximum storage (stage1Input storage s X N B) j ∧
        t.mem Y (s.pids 0) = .mk (channel storage).toTileDType
          (ofFloat (channel storage) (stage1Input storage s X N B j)) ∧
        t.mem (Region.cast I) (s.pids 0) = .mk .int (Int.ofNat (j.val + s.pids 0 * B)) ∧
        isNaN storage (stage1Input storage s X N B j) = Bool.false) ∧
      (∀ r o, (r ≠ Y ∨ o ≠ s.pids 0) → (r ≠ Region.cast I ∨ o ≠ s.pids 0) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := stage1_run storage rest s X Y I N B index64 hB
  refine ⟨t, he, ?_, ?_⟩
  · refine ⟨stage1Selected storage s X N B hB,
      ⟨best_maximal storage _ hB, best_leftmost storage _ hB⟩, ?_, ?_, hnotNaN _⟩
    · rw [hm]
      unfold stage1Memory
      rw [State.write_other _ _ _ _ _ _ (Or.inl hYI)]
      exact State.write_same _ _ _ _
    · rw [hm]
      exact State.write_same _ _ _ _
  · intro r o hY hI
    rw [hm]
    unfold stage1Memory
    rw [State.write_other _ _ _ _ _ _ hI, State.write_other _ _ _ _ _ _ hY]

/-- Complete second-stage typed source: the output is the initial index payload
at the independently maximal padded input lane. No conclusion or successful
execution is assumed. The contract preserves all other tagged memory cells.
Non-NaN operands are the stated ordered-comparison domain; no NaN propagation
policy or compiled reduction-tree behavior is inferred from this theorem. -/
theorem argmax_stage2_bit_correctness
    (storage : Storage) (rest : Algebra (Bits storage))
    (s : State (Bits storage)) (X : RegionName) (I O : Region .int)
    (N B : Nat) (hB : 0 < B)
    (hnotNaN : ∀ i, isNaN storage (stage2Input storage s X N B i) = Bool.false) :
    ∃ t, FP.Structural.exec (BitBackend.backend storage rest) (stage2 storage X I O N B) s = some t ∧
      (∃ j : Fin B, IsArgMaximum storage (stage2Input storage s X N B) j ∧
        t.mem (Region.cast O) 0 = .mk .int ((s.mem (Region.cast I) j.val).read .int) ∧
        isNaN storage (stage2Input storage s X N B j) = Bool.false) ∧
      (∀ r o, r ≠ Region.cast O ∨ o ≠ 0 → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := stage2_run storage rest s X I O N B hB
  refine ⟨t, he, ?_, ?_⟩
  · refine ⟨stage2Selected storage s X N B hB, ?_, ?_, hnotNaN _⟩
    · exact ⟨best_maximal storage _ hB, best_leftmost storage _ hB⟩
    · rw [hm]
      exact State.write_same _ _ _ _
  · intro r o hmiss
    rw [hm]
    exact State.write_other _ _ _ _ _ _ hmiss

#axiomsClean stage1_run
#axiomsClean argmax_stage1_bit_correctness
#axiomsClean stage2_run
#axiomsClean argmax_stage2_bit_correctness

end
end VeriTile.Bench.TritonBenchG.TritonArgmax.BitCorrect

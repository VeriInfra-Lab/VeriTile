import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHReadback
import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelRealModel
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.RealReference

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
namespace H
open ForwardHSource

def xs (c : Shape) (seed : Bool) (s : State ℝ) := inputs c s.mem seed Bool.false

theorem key_load (c : Shape) (seed : Bool) (s : State ℝ) (n : Nat) (i : TileIndex [c.BK,64]) :
    keyLoad algebra c s n i=
      kAt realOps c (xs c seed s) (s.pids 2) (n*64+i.2.1.val) (feature c s i.1.val) := by
  simp [keyLoad,kAt,masked,xs,inputs,algebra,realOps,zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]

theorem value_load (c : Shape) (seed : Bool) (s : State ℝ) (n : Nat) (i : TileIndex [64,c.BV]) :
    valueLoad algebra c s n i=
      vAt realOps c (xs c seed s) (s.pids 2) (n*64+i.1.val) (column c s i.2.1.val) := by
  simp [valueLoad,vAt,masked,xs,inputs,algebra,realOps,zero]

theorem block_decay (c : Shape) (s : State ℝ) (n : Nat) :
    blockDecay algebra c s n=Whole001.decay realOps c (s.pids 2) n := by
  unfold blockDecay
  rw [rate_eq]
  by_cases hn : n=64
  · subst n
    rfl
  · simp only [hn,↓reduceIte]
    rfl

theorem time_decay (c : Shape) (s : State ℝ) (n : Nat) (i : TileIndex [64]) :
    timeDecay algebra c s n i=Whole001.decay realOps c (s.pids 2) ((n:Int)-i.1.val-1) :=
  decay_eq c (s.pids 2) _

theorem initial_reference (c : Shape) (seed : Bool) (s : State ℝ) (i : Index c) :
    initValue algebra c seed s i=
      Reference.history realOps c (xs c seed s) (s.pids 2) 0
        (feature c s i.1.val) (column c s i.2.1.val) := by
  cases seed <;> simp [initValue,Reference.history,masked,xs,inputs,algebra,realOps,
    zero,active,stateAddress]

theorem update_body (c : Shape) (seed : Bool) (s : State ℝ) (n : Nat)
    (carry : Index c → ℝ) (ref : Tile ℝ) (i : Index c) (hn : n<c.chunks)
    (hi : carry i=ref (feature c s i.1.val) (column c s i.2.1.val)) :
    ForwardHSource.update algebra dot c s n carry i=
      Body.nextH realOps c (xs c seed s) (s.pids 2) n ref
        (feature c s i.1.val) (column c s i.2.1.val) := by
  change blockDecay algebra c s (actualLength c n)*carry i+
    (∑ t : Fin 64,keyLoad algebra c s n ⟨i.1,t,()⟩ * gatedValue algebra c s n ⟨t,i.2.1,()⟩)=_
  rw [block_decay,hi]
  have hl : actualLength c n=c.length n := (c.length_source n hn).symm
  rw [hl]
  unfold Body.nextH
  rw [RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  rw [key_load c seed s]
  unfold gatedValue
  rw [value_load c seed s,time_decay,hl]
  rfl

theorem trajectory_reference (c : Shape) (seed : Bool) (s : State ℝ) (n : Nat)
    (hn : n≤c.chunks) (i : Index c) :
    (trajectory algebra dot c seed s n).1 i=
      Reference.history realOps c (xs c seed s) (s.pids 2) n
        (feature c s i.1.val) (column c s i.2.1.val) := by
  induction n with
  | zero => exact initial_reference c seed s i
  | succ n ih =>
    change ForwardHSource.update algebra dot c s n (trajectory algebra dot c seed s n).1 i=_
    exact update_body c seed s n _ (Reference.history realOps c (xs c seed s) (s.pids 2) n)
      i (by omega) (ih (by omega))

theorem history_reference (c : Shape) (seed final : Bool) (s : State ℝ)
    (n : Nat) (i : Index c) (hn : n<c.chunks) (hi : active c s i) :
    ((written algebra dot c seed final s).mem "h" (historyAddress c s n i)).read .real=
      Reference.history realOps c (xs c seed s) (s.pids 2) n
        (feature c s i.1.val) (column c s i.2.1.val) := by
  rw [history_readback algebra dot c seed final s n i hn hi]
  exact trajectory_reference c seed s n (Nat.le_of_lt hn) i

theorem final_reference (c : Shape) (seed : Bool) (s : State ℝ)
    (i : Index c) (hi : active c s i) :
    ((written algebra dot c seed Bool.true s).mem "ht" (stateAddress c s i)).read .real=
      Reference.finalState realOps c (xs c seed s) (s.pids 2)
        (feature c s i.1.val) (column c s i.2.1.val) := by
  rw [final_readback algebra dot c seed s i hi]
  exact trajectory_reference c seed s c.chunks le_rfl i

theorem history_math (c : Shape) (seed final : Bool) (s : State ℝ)
    (n : Nat) (i : Index c) (hn : n<c.chunks) (hi : active c s i) :
    ((written algebra dot c seed final s).mem "h" (historyAddress c s n i)).read .real=
      MathTarget.state c (xs c seed s) Bool.true (s.pids 2) (n*64)
        (feature c s i.1.val) (column c s i.2.1.val) := by
  rw [history_reference c seed final s n i hn hi,
    RealScalar.history_scalar c (xs c seed s) (s.pids 2) n _ _ (Nat.le_of_lt hn) hi.1 hi.2,
    ScalarSource.history_correct c (xs c seed s) Bool.true (s.pids 2) n _ _ hn]

theorem final_math (c : Shape) (seed : Bool) (s : State ℝ)
    (i : Index c) (hi : active c s i) :
    ((written algebra dot c seed Bool.true s).mem "ht" (stateAddress c s i)).read .real=
      MathTarget.finalState c (xs c seed s) Bool.true (s.pids 2)
        (feature c s i.1.val) (column c s i.2.1.val) := by
  rw [final_reference c seed s i hi,
    RealScalar.reference_final c (xs c seed s) (s.pids 2) _ _ hi.1 hi.2]

/-- The selected quotation executes from arbitrary Real input cells. All
history cells and the optional terminal state are the independent retention
sums, and every cell outside this program's active stores is preserved. -/
theorem source_result (c : Shape) (seed final : Bool) (s : State ℝ) :
    ∃ t, FP.Structural.exec (engine (withDot algebra dot)) (kernel c seed final) s=some t ∧
      (∀ n i, n<c.chunks → active c s i →
        (t.mem "h" (historyAddress c s n i)).read .real=
          MathTarget.state c (xs c seed s) Bool.true (s.pids 2) (n*64)
            (feature c s i.1.val) (column c s i.2.1.val)) ∧
      (final=Bool.true → ∀ i, active c s i →
        (t.mem "ht" (stateAddress c s i)).read .real=
          MathTarget.finalState c (xs c seed s) Bool.true (s.pids 2)
            (feature c s i.1.val) (column c s i.2.1.val)) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"h" ∨ ∀ n, n<c.chunks → ∀ i, active c s i → off≠historyAddress c s n i) →
        (final=Bool.false ∨ r≠"ht" ∨ ∀ i, active c s i → off≠stateAddress c s i) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory algebra dot c seed final s
  refine ⟨t,ht,?_,?_,?_⟩
  · intro n i hn hi
    rw [hm]
    exact history_math c seed final s n i hn hi
  · intro hf i hi
    subst final
    rw [hm]
    exact final_math c seed s i hi
  · intro r off hh hf
    rw [hm]
    exact written_frame algebra dot c seed final s r off hh hf

#axiomsClean trajectory_reference
#axiomsClean history_reference
#axiomsClean final_reference
#axiomsClean source_result
end H
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal

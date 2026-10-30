import bench.tritonbench_g.chunk_retention.Sources.BackwardHComplete
import bench.tritonbench_g.chunk_retention.Sources.ForwardHReal

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal.BH
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel BackwardHSource
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def xs (c : Shape) (terminal : Bool) (s : State ℝ) := inputs c s.mem Bool.false terminal

theorem query_load (c : Shape) (terminal : Bool) (s : State ℝ) (n : Nat) (i : TileIndex [c.BK,64]) :
    queryLoad algebra c s n i=
      qAt realOps c (xs c terminal s) (s.pids 2) (chunk c n*64+i.2.1.val) (feature c s i.1.val) := by
  simp [queryLoad,qAt,masked,xs,inputs,algebra,realOps,zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem incoming_load (c : Shape) (terminal : Bool) (s : State ℝ) (n : Nat) (i : TileIndex [64,c.BV]) :
    incomingLoad algebra c s n i=
      gAt realOps c (xs c terminal s) (s.pids 2) (chunk c n*64+i.1.val) (column c s i.2.1.val) := by
  simp [incomingLoad,gAt,masked,xs,inputs,algebra,realOps,zero]
theorem time_decay (c : Shape) (s : State ℝ) (i : TileIndex [64]) :
    timeDecay algebra c s i=Whole001.decay realOps c (s.pids 2) (i.1.val+1) :=
  decay_eq c (s.pids 2) _
theorem initial_reference (c : Shape) (terminal : Bool) (s : State ℝ) (scale : ℝ) (i : Index c) :
    initValue algebra c terminal s i=
      Reference.reverseCarry realOps c (xs c terminal s) scale (s.pids 2) 0
        (feature c s i.1.val) (column c s i.2.1.val) := by
  cases terminal <;> simp [initValue,Reference.reverseCarry,masked,xs,inputs,algebra,realOps,
    zero,active,stateAddress,ForwardHSource.active,ForwardHSource.stateAddress]

theorem update_body (c : Shape) (terminal : Bool) (s : State ℝ) (scale : ℝ) (n : Nat)
    (carry : Index c → ℝ) (ref : Tile ℝ) (i : Index c) (hn : n<c.chunks)
    (hi : carry i=ref (feature c s i.1.val) (column c s i.2.1.val)) :
    BackwardHSource.update algebra dot c s scale n carry i=
      Body.nextDH realOps c (xs c terminal s) scale (s.pids 2) (chunk c n) ref
        (feature c s i.1.val) (column c s i.2.1.val) := by
  change blockDecay algebra c s (actualLength c (chunk c n))*carry i+
    (∑ t : Fin 64,queryLoad algebra c s n ⟨i.1,t,()⟩ * gatedIncoming algebra c s scale n ⟨t,i.2.1,()⟩)=_
  simp only [BackwardHSource.blockDecay]
  rw [H.block_decay,hi]
  have hc : chunk c n<c.chunks := by dsimp [chunk]; omega
  have hl : actualLength c (chunk c n)=c.length (chunk c n) := (c.length_source _ hc).symm
  rw [hl]
  unfold Body.nextDH
  rw [RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  rw [query_load c terminal s]
  unfold gatedIncoming
  rw [incoming_load c terminal s,time_decay]
  rfl

theorem trajectory_reference (c : Shape) (terminal : Bool) (s : State ℝ) (scale : ℝ)
    (n : Nat) (hn : n≤c.chunks) (i : Index c) :
    (trajectory algebra dot c s scale terminal n).1 i=
      Reference.reverseCarry realOps c (xs c terminal s) scale (s.pids 2) n
        (feature c s i.1.val) (column c s i.2.1.val) := by
  induction n with
  | zero => exact initial_reference c terminal s scale i
  | succ n ih =>
    change BackwardHSource.update algebra dot c s scale n (trajectory algebra dot c s scale terminal n).1 i=_
    exact update_body c terminal s scale n _
      (Reference.reverseCarry realOps c (xs c terminal s) scale (s.pids 2) n) i (by omega) (ih (by omega))

theorem history_math (c : Shape) (terminal : Bool) (s : State ℝ) (scale : ℝ)
    (n : Nat) (i : Index c) (hn : n<c.chunks) (hi : active c s i) :
    ((written algebra dot c s scale terminal).mem "dh" (historyAddress c s (chunk c n) i)).read .real=
      MathTarget.future c (xs c terminal s) scale (s.pids 2) (c.boundary (c.chunks-n))
        (feature c s i.1.val) (column c s i.2.1.val) := by
  rw [history_readback algebra dot c s scale terminal n i hn hi]
  change (trajectory algebra dot c s scale terminal n).1 i=_
  rw [trajectory_reference c terminal s scale n (Nat.le_of_lt hn),
    RealScalar.reverse_scalar c (xs c terminal s) scale (s.pids 2) n _ _ (Nat.le_of_lt hn) hi.1 hi.2,
    ScalarSource.reverse_carry_correct c (xs c terminal s) scale (s.pids 2) n (Nat.le_of_lt hn)]

theorem source_result (c : Shape) (terminal : Bool) (s : State ℝ) (scale : ℝ) :
    ∃ t, FP.Structural.exec (engine (withDot algebra dot)) (kernel c scale terminal) s=some t ∧
      (∀ n i, n<c.chunks → active c s i →
        (t.mem "dh" (historyAddress c s (chunk c n) i)).read .real=
          MathTarget.future c (xs c terminal s) scale (s.pids 2) (c.boundary (c.chunks-n))
            (feature c s i.1.val) (column c s i.2.1.val)) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"dh" ∨ ∀ n, n<c.chunks → ∀ i, active c s i → off≠historyAddress c s n i) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory algebra dot c scale terminal s
  refine ⟨t,ht,?_,?_⟩
  · intro n i hn hi
    rw [hm]
    exact history_math c terminal s scale n i hn hi
  · intro r off hf
    rw [hm]
    exact frame algebra dot c s scale terminal r off hf

#axiomsClean trajectory_reference
#axiomsClean source_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal.BH

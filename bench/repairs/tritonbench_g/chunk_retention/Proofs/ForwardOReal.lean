import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardOComplete
import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelRealModel

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal
open VeriTile Triton FP.Structural KernelModel
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
namespace O
open ForwardOSource

def history (c : Shape) (s : State ℝ) (e j : Nat) : ℝ :=
  (s.mem "h" (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+e*c.V+j)).read .real
def xs (c : Shape) (s : State ℝ) := inputs c s.mem Bool.false Bool.false

theorem query_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [64,c.BK]) :
    featureLoad algebra c s "q" f i=
      qAt realOps c (xs c s) (s.pids 2) (row s i.1.val) (feature c f i.2.1.val) := by
  simp [featureLoad,qAt,masked,xs,inputs,algebra,realOps,zero]
theorem key_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [c.BK,64]) :
    keyLoad algebra c s f i=
      kAt realOps c (xs c s) (s.pids 2) (row s i.2.1.val) (feature c f i.1.val) := by
  simp [keyLoad,kAt,masked,xs,inputs,algebra,realOps,zero,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem history_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [c.BK,c.BV]) :
    historyLoad algebra c s f i=
      masked realOps c.K c.V (history c s) (feature c f i.1.val) (column c s i.2.1.val) := by
  simp [historyLoad,history,masked,algebra,realOps,zero]
theorem time_decay (c : Shape) (s : State ℝ) (i : TileIndex [64]) :
    rowDecay algebra c s i=Whole001.decay realOps c (s.pids 2) (i.1.val+1) := by
  simp [rowDecay,decay_eq]
theorem score_decay (c : Shape) (s : State ℝ) (i : ScoreIndex) :
    scoreDecay algebra c s i=causalWeight realOps c (s.pids 2) i.1.val i.2.1.val := by
  simp only [scoreDecay,causalWeight,decay_eq]
  rfl

def carryReference (c : Shape) (s : State ℝ) (n : Nat) (i : Index c) : ℝ :=
  blockChain realOps c.BK
    (fun e => qAt realOps c (xs c s) (s.pids 2) (row s i.1.val) e *
      Whole001.decay realOps c (s.pids 2) (i.1.val+1))
    (fun e => masked realOps c.K c.V (history c s) e (column c s i.2.1.val)) n 0
def scoreReference (c : Shape) (s : State ℝ) (n : Nat) (i : ScoreIndex) : ℝ :=
  blockChain realOps c.BK
    (fun e => qAt realOps c (xs c s) (s.pids 2) (row s i.1.val) e)
    (fun e => kAt realOps c (xs c s) (s.pids 2) (row s i.2.1.val) e) n 0
def states (c : Shape) (s : State ℝ) := KernelFold.sequenceValue (ForwardOSource.update algebra dot c s)
  (⟨fun _ => (0:ℝ),fun _ => (0:ℝ)⟩ : Accum c ℝ)

theorem states_carry (c : Shape) (s : State ℝ) (n : Nat) (i : Index c) :
    (states c s n).1 i=carryReference c s n i := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (states c s n).1 i +
      (∑ e : Fin c.BK,queryGate algebra c s n ⟨i.1,e,()⟩ *
        historyLoad algebra c s n ⟨e,i.2.1,()⟩)=_
    rw [ih]
    unfold carryReference
    rw [blockChain,RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
    congr 1
    apply Finset.sum_congr rfl
    intro e _
    simp only [queryGate]
    rw [query_load,history_load,time_decay]
    rfl

theorem states_score (c : Shape) (s : State ℝ) (n : Nat) (i : ScoreIndex) :
    (states c s n).2 i=scoreReference c s n i := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (states c s n).2 i +
      (∑ e : Fin c.BK,featureLoad algebra c s "q" n ⟨i.1,e,()⟩ *
        keyLoad algebra c s n ⟨e,i.2.1,()⟩)=_
    rw [ih]
    unfold scoreReference
    rw [blockChain,RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
    congr 1
    apply Finset.sum_congr rfl
    intro e _
    rw [query_load,key_load]
    rfl

theorem accumulated_carry (c : Shape) (s : State ℝ) (i : Index c) :
    (accumulated algebra dot c s).1 i=
      Body.outputCarry realOps c (xs c s) (s.pids 2) (s.pids 1) i.1.val
        (column c s i.2.1.val) (history c s) := by
  change (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.foldRange
    (ForwardOSource.update algebra dot c s) 0 c.keyBlocks 1
      (⟨fun _ => (0:ℝ),fun _ => (0:ℝ)⟩ : Accum c ℝ)).1 i=_
  rw [KernelFold.from_zero]
  exact states_carry c s c.keyBlocks i
theorem accumulated_score (c : Shape) (s : State ℝ) (i : ScoreIndex) :
    (accumulated algebra dot c s).2 i=
      Body.score realOps c (xs c s) (s.pids 2) (s.pids 1) i.1.val i.2.1.val := by
  change (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.foldRange
    (ForwardOSource.update algebra dot c s) 0 c.keyBlocks 1
      (⟨fun _ => (0:ℝ),fun _ => (0:ℝ)⟩ : Accum c ℝ)).2 i=_
  rw [KernelFold.from_zero]
  exact states_score c s c.keyBlocks i

theorem value_load (c : Shape) (s : State ℝ) (i : Index c) :
    valueLoad algebra c s i=vAt realOps c (xs c s) (s.pids 2) (row s i.1.val) (column c s i.2.1.val) := by
  simp [valueLoad,active,outputAddress,vAt,masked,algebra,realOps,zero,xs,inputs]
theorem attention_eq (c : Shape) (s : State ℝ) (i : ScoreIndex) :
    attention algebra dot c s i=
      Body.score realOps c (xs c s) (s.pids 2) (s.pids 1) i.1.val i.2.1.val *
        causalWeight realOps c (s.pids 2) i.1.val i.2.1.val := by
  unfold attention
  rw [accumulated_score,score_decay]
  rfl

theorem value_body (c : Shape) (s : State ℝ) (scale : ℝ) (i : Index c) :
    value algebra dot c scale s i=
      Body.output realOps c (xs c s) scale (s.pids 2) (s.pids 1)
        (history c s) i.1.val (column c s i.2.1.val) := by
  change ((accumulated algebra dot c s).1 i+
    (∑ t : Fin 64,attention algebra dot c s ⟨i.1,t,()⟩ * valueLoad algebra c s ⟨t,i.2.1,()⟩))*scale=_
  rw [accumulated_carry]
  unfold Body.output
  simp only [RealScalar.real_cast,RealScalar.real_mul]
  rw [RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
  congr 2
  apply Finset.sum_congr rfl
  intro t _
  rw [attention_eq,value_load]
  rfl

#specNonCircular history avoiding [run,FP.Structural.exec]
#axiomsClean states_carry
#axiomsClean states_score
#axiomsClean value_body
end O
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal

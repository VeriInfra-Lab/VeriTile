import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardReal
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.RealBody
import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelFold
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
attribute [local simp] KernelOutput.shape
set_option linter.unusedSimpArgs false
namespace O
open VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
abbrev SharedShape := VeriTile.Bench.Repairs.ChunkRetention.Whole001.Shape
abbrev shared := KernelOutput.shape
abbrev dot := KernelOutput.sharedDot KernelReal.dotPrimitive

def history (c : Shape) (s : State ℝ) (e j : Nat) : ℝ :=
  (s.mem "h" (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+e*c.V+j)).read .real
def xs (c : Shape) (s : State ℝ) := inputs c s.mem Bool.false Bool.false

theorem query_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [64,c.BK]) :
    featureLoad algebra (shared c) s "q" f i=
      qAt realOps c (xs c s) (s.pids 2) (row s i.1.val) (feature (shared c) f i.2.1.val) := by
  simp [featureLoad,qAt,masked,xs,inputs,algebra,realOps,VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.zero]
theorem key_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [c.BK,64]) :
    keyLoad algebra (shared c) s f i=
      kAt realOps c (xs c s) (s.pids 2) (row s i.2.1.val) (feature (shared c) f i.1.val) := by
  simp [keyLoad,kAt,masked,xs,inputs,algebra,realOps,VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.zero,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem history_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [c.BK,c.BV]) :
    historyLoad algebra (shared c) s f i=
      masked realOps c.K c.V (history c s) (feature (shared c) f i.1.val) (column (shared c) s i.2.1.val) := by
  simp [historyLoad,history,masked,algebra,realOps,VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.zero]
  rfl
theorem time_decay (c : Shape) (s : State ℝ) (i : TileIndex [64]) :
    rowDecay algebra (shared c) s i=VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.decay realOps c (s.pids 2) (i.1.val+1) := by
  simp only [rowDecay]
  change algebra.unary (some .fp32) .exp2
    (algebra.binary (some .fp32) .real .mul (algebra.fromInt (some .fp32) (i.1.val+1))
      (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.rate algebra (shared c) (s.pids 2)))=_
  rw [shared_rate_eq c ⟨s.pids 0,s.pids 1,s.pids 2⟩]
  rfl
theorem score_decay (c : Shape) (s : State ℝ) (i : ScoreIndex) :
    scoreDecay algebra (shared c) s i=causalWeight realOps c (s.pids 2) i.1.val i.2.1.val := by
  unfold scoreDecay VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.decay
  rw [shared_rate_eq c ⟨s.pids 0,s.pids 1,s.pids 2⟩]
  rfl

def carryReference (c : Shape) (s : State ℝ) (n : Nat) (i : Index (shared c)) : ℝ :=
  blockChain realOps c.BK
    (fun e => qAt realOps c (xs c s) (s.pids 2) (row s i.1.val) e *
      VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.decay realOps c (s.pids 2) (i.1.val+1))
    (fun e => masked realOps c.K c.V (history c s) e (column (shared c) s i.2.1.val)) n 0
def scoreReference (c : Shape) (s : State ℝ) (n : Nat) (i : ScoreIndex) : ℝ :=
  blockChain realOps c.BK
    (fun e => qAt realOps c (xs c s) (s.pids 2) (row s i.1.val) e)
    (fun e => kAt realOps c (xs c s) (s.pids 2) (row s i.2.1.val) e) n 0
def states (c : Shape) (s : State ℝ) := VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelFold.sequenceValue (VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.update algebra dot (shared c) s)
  (⟨fun _ => (0:ℝ),fun _ => (0:ℝ)⟩ : Accum (shared c) ℝ)

theorem states_carry (c : Shape) (s : State ℝ) (n : Nat) (i : Index (shared c)) :
    (states c s n).1 i=carryReference c s n i := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (states c s n).1 i +
      (∑ e : Fin c.BK,queryGate algebra (shared c) s n ⟨i.1,e,()⟩ *
        historyLoad algebra (shared c) s n ⟨e,i.2.1,()⟩)=_
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
      (∑ e : Fin c.BK,featureLoad algebra (shared c) s "q" n ⟨i.1,e,()⟩ *
        keyLoad algebra (shared c) s n ⟨e,i.2.1,()⟩)=_
    rw [ih]
    unfold scoreReference
    rw [blockChain,RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
    congr 1
    apply Finset.sum_congr rfl
    intro e _
    rw [query_load,key_load]
    rfl

theorem accumulated_carry (c : Shape) (s : State ℝ) (i : Index (shared c)) :
    (accumulated algebra dot (shared c) s).1 i=
      Body.outputCarry realOps c (xs c s) (s.pids 2) (s.pids 1) i.1.val
        (column (shared c) s i.2.1.val) (history c s) := by
  change (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.foldRange
    (VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.update algebra dot (shared c) s) 0 c.keyBlocks 1
      (⟨fun _ => (0:ℝ),fun _ => (0:ℝ)⟩ : Accum (shared c) ℝ)).1 i=_
  rw [VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelFold.from_zero]
  exact states_carry c s c.keyBlocks i
theorem accumulated_score (c : Shape) (s : State ℝ) (i : ScoreIndex) :
    (accumulated algebra dot (shared c) s).2 i=
      Body.score realOps c (xs c s) (s.pids 2) (s.pids 1) i.1.val i.2.1.val := by
  change (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.foldRange
    (VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.update algebra dot (shared c) s) 0 c.keyBlocks 1
      (⟨fun _ => (0:ℝ),fun _ => (0:ℝ)⟩ : Accum (shared c) ℝ)).2 i=_
  rw [VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelFold.from_zero]
  exact states_score c s c.keyBlocks i

theorem value_load (c : Shape) (s : State ℝ) (i : Index (shared c)) :
    valueLoad algebra (shared c) s i=vAt realOps c (xs c s) (s.pids 2) (row s i.1.val) (column (shared c) s i.2.1.val) := by
  simp [valueLoad,active,outputAddress,vAt,masked,algebra,realOps,VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.zero,xs,inputs]
theorem attention_eq (c : Shape) (s : State ℝ) (i : ScoreIndex) :
    attention algebra dot (shared c) s i=
      Body.score realOps c (xs c s) (s.pids 2) (s.pids 1) i.1.val i.2.1.val *
        causalWeight realOps c (s.pids 2) i.1.val i.2.1.val := by
  unfold attention
  rw [accumulated_score,score_decay]
  rfl

theorem value_body (c : Shape) (s : State ℝ) (scale : ℝ) (i : Index (shared c)) :
    value algebra dot (shared c) scale s i=
      Body.output realOps c (xs c s) scale (s.pids 2) (s.pids 1)
        (history c s) i.1.val (column (shared c) s i.2.1.val) := by
  change ((accumulated algebra dot (shared c) s).1 i+
    (∑ t : Fin 64,attention algebra dot (shared c) s ⟨i.1,t,()⟩ * valueLoad algebra (shared c) s ⟨t,i.2.1,()⟩))*scale=_
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
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal

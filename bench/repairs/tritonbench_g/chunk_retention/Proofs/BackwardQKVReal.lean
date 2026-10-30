import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVReadback
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHReal

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal.QKV
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel BackwardQKVSource
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

def xs (c : Shape) (s : State ℝ) := inputs c s.mem Bool.false Bool.false
def history (c : Shape) (s : State ℝ) (r : RegionName) (e j : Nat) : ℝ :=
  (s.mem r (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+e*c.V+j)).read .real
theorem query_load (c : Shape) (s : State ℝ) (i : TileIndex [c.BK,64]) :
    queryLoad algebra c s i=qAt realOps c (xs c s) (s.pids 2) (row s i.2.1.val) (feature c s i.1.val) := by
  simp [queryLoad,qAt,masked,xs,inputs,algebra,realOps,zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem key_load (c : Shape) (s : State ℝ) (i : KeyIndex c) :
    keyLoad algebra c s i=kAt realOps c (xs c s) (s.pids 2) (row s i.1.val) (feature c s i.2.1.val) := by
  simp [keyLoad,kAt,masked,xs,inputs,algebra,realOps,zero]
theorem value_load (c : Shape) (s : State ℝ) (f : Nat) (i : ValueIndex c) :
    valueLoad algebra c s "v" f i=vAt realOps c (xs c s) (s.pids 2) (row s i.1.val) (column c f i.2.1.val) := by
  simp [valueLoad,vAt,masked,xs,inputs,algebra,realOps,zero]
theorem incoming_load (c : Shape) (s : State ℝ) (f : Nat) (i : ValueIndex c) :
    valueLoad algebra c s "incoming" f i=gAt realOps c (xs c s) (s.pids 2) (row s i.1.val) (column c f i.2.1.val) := by
  simp [valueLoad,gAt,masked,xs,inputs,algebra,realOps,zero]
theorem history_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [c.BV,c.BK]) :
    historyLoad algebra c s f i=
      masked realOps c.K c.V (history c s "h") (feature c s i.2.1.val) (column c f i.1.val) := by
  simp [historyLoad,history,masked,algebra,realOps,zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem dhistory_load (c : Shape) (s : State ℝ) (f : Nat) (i : TileIndex [c.BK,c.BV]) :
    dhistoryLoad algebra c s f i=
      masked realOps c.K c.V (history c s "dh") (feature c s i.1.val) (column c f i.2.1.val) := by
  simp [dhistoryLoad,history,masked,algebra,realOps,zero]
theorem query_decay (c : Shape) (s : State ℝ) (scale : ℝ) (i : TileIndex [64]) :
    queryDecay algebra c s scale i=Whole001.decay realOps c (s.pids 2) (i.1.val+1)*scale := by
  unfold queryDecay
  rw [decay_eq]
  rfl
theorem key_decay (c : Shape) (s : State ℝ) (i : TileIndex [64]) (hc : s.pids 1<c.chunks) :
    keyDecay algebra c s i=Whole001.decay realOps c (s.pids 2) ((c.length (s.pids 1):Int)-i.1.val-1) := by
  unfold keyDecay
  rw [decay_eq,show ForwardHSource.actualLength c (s.pids 1)=c.length (s.pids 1) from (c.length_source _ hc).symm]
theorem score_decay (c : Shape) (s : State ℝ) (scale : ℝ) (i : ScoreIndex) :
    scoreDecay algebra c s scale i=causalWeight realOps c (s.pids 2) i.1.val i.2.1.val*scale := by
  simp only [scoreDecay,decay_eq,causalWeight]
  rfl

def queryReference (c : Shape) (s : State ℝ) (n : Nat) (i : KeyIndex c) : ℝ :=
  blockChain realOps c.BV
    (fun j => gAt realOps c (xs c s) (s.pids 2) (row s i.1.val) j)
    (fun j => masked realOps c.K c.V (history c s "h") (feature c s i.2.1.val) j) n 0
def keyReference (c : Shape) (s : State ℝ) (n : Nat) (i : KeyIndex c) : ℝ :=
  blockChain realOps c.BV
    (fun j => vAt realOps c (xs c s) (s.pids 2) (row s i.1.val) j)
    (fun j => masked realOps c.K c.V (history c s "dh") (feature c s i.2.1.val) j) n 0
def scoreReference (c : Shape) (s : State ℝ) (n : Nat) (i : ScoreIndex) : ℝ :=
  blockChain realOps c.BV
    (fun j => gAt realOps c (xs c s) (s.pids 2) (row s i.1.val) j)
    (fun j => vAt realOps c (xs c s) (s.pids 2) (row s i.2.1.val) j) n 0

theorem trajectory_query (c : Shape) (s : State ℝ) (scale : ℝ) (n : Nat) (i : KeyIndex c) :
    (trajectory algebra dot c s scale n).1.1 i=queryReference c s n i := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (trajectory algebra dot c s scale n).1.1 i+
      (∑ j : Fin c.BV,valueLoad algebra c s "incoming" n ⟨i.1,j,()⟩ * historyLoad algebra c s n ⟨j,i.2.1,()⟩)=_
    rw [ih]
    unfold queryReference
    rw [blockChain,RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
    congr 1
    apply Finset.sum_congr rfl
    intro j _
    rw [incoming_load,history_load]
    rfl
theorem trajectory_key (c : Shape) (s : State ℝ) (scale : ℝ) (n : Nat) (i : KeyIndex c) :
    (trajectory algebra dot c s scale n).1.2.1 i=keyReference c s n i := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (trajectory algebra dot c s scale n).1.2.1 i+
      (∑ j : Fin c.BV,valueLoad algebra c s "v" n ⟨i.1,j,()⟩ * dhistoryLoad algebra c s n ⟨i.2.1,j,()⟩)=_
    rw [ih]
    unfold keyReference
    rw [blockChain,RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
    congr 1
theorem trajectory_score (c : Shape) (s : State ℝ) (scale : ℝ) (n : Nat) (i : ScoreIndex) :
    (trajectory algebra dot c s scale n).1.2.2 i=scoreReference c s n i := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (trajectory algebra dot c s scale n).1.2.2 i+
      (∑ j : Fin c.BV,valueLoad algebra c s "incoming" n ⟨i.1,j,()⟩ * valueLoad algebra c s "v" n ⟨i.2.1,j,()⟩)=_
    rw [ih]
    unfold scoreReference
    rw [blockChain,RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
    congr 1

theorem score_gradient (c : Shape) (s : State ℝ) (scale : ℝ) (i : ScoreIndex) :
    scoreGradient algebra c s scale (trajectory algebra dot c s scale c.valueBlocks).1 i=
      Body.ds realOps c (xs c s) scale (s.pids 2) (s.pids 1) i.1.val i.2.1.val := by
  unfold scoreGradient
  rw [trajectory_score,score_decay]
  rfl

theorem query_body (c : Shape) (s : State ℝ) (scale : ℝ) (i : KeyIndex c) :
    queryValue algebra dot c s scale (trajectory algebra dot c s scale c.valueBlocks).1 i=
      Body.dq realOps c (xs c s) scale (s.pids 2) (s.pids 1) (history c s "h")
        i.1.val (feature c s i.2.1.val) := by
  change (trajectory algebra dot c s scale c.valueBlocks).1.1 i*queryDecay algebra c s scale ⟨i.1,()⟩+
    (∑ t : Fin 64,scoreGradient algebra c s scale (trajectory algebra dot c s scale c.valueBlocks).1 ⟨i.1,t,()⟩ *
      keyLoad algebra c s ⟨t,i.2.1,()⟩)=_
  rw [trajectory_query,query_decay]
  unfold Body.dq
  simp only [RealScalar.real_cast,RealScalar.real_mul]
  rw [RealScalar.chain_sum,←Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  rw [score_gradient,key_load]
  rfl

theorem key_body (c : Shape) (s : State ℝ) (scale : ℝ) (i : KeyIndex c) (hc : s.pids 1<c.chunks) :
    keyValue algebra dot c s scale (trajectory algebra dot c s scale c.valueBlocks).1 i=
      Body.dk realOps c (xs c s) scale (s.pids 2) (s.pids 1) (history c s "dh")
        i.1.val (feature c s i.2.1.val) := by
  change (trajectory algebra dot c s scale c.valueBlocks).1.2.1 i*keyDecay algebra c s ⟨i.1,()⟩+
    (∑ t : Fin 64,queryLoad algebra c s ⟨i.2.1,t,()⟩ *
      scoreGradient algebra c s scale (trajectory algebra dot c s scale c.valueBlocks).1 ⟨t,i.1,()⟩)=_
  rw [trajectory_key,key_decay c s _ hc]
  unfold Body.dk
  simp only [RealScalar.real_cast,RealScalar.real_mul,RealScalar.real_add,RealScalar.real_zero]
  rw [RealScalar.chain_sum,zero_add,←Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  rw [query_load,score_gradient]
  rfl

theorem partial_score (c : Shape) (s : State ℝ) (scale : ℝ) (i : ScoreIndex) :
    partialScore algebra dot c s scale i=
      Body.partialScore realOps c (xs c s) scale (s.pids 0) (s.pids 2) (s.pids 1) i.1.val i.2.1.val := by
  change (∑ e : Fin c.BK,keyLoad algebra c s ⟨i.1,e,()⟩ * queryLoad algebra c s ⟨e,i.2.1,()⟩)*
    scoreDecay algebra c s scale ⟨i.2.1,i.1,()⟩=_
  rw [score_decay]
  unfold Body.partialScore
  simp only [RealScalar.real_mul,RealScalar.real_zero]
  rw [RealScalar.chain_sum,zero_add,←Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro e _
  rw [key_load,query_load]
  rfl

theorem partial_body (c : Shape) (s : State ℝ) (scale : ℝ) (f : Nat) (i : ValueIndex c)
    (hc : s.pids 1<c.chunks) :
    partialValue algebra dot c s scale f i=
      Body.dvPartial realOps c (xs c s) scale (s.pids 0) (s.pids 2) (s.pids 1)
        (history c s "dh") i.1.val (column c f i.2.1.val) := by
  change (∑ e : Fin c.BK,keyLoad algebra c s ⟨i.1,e,()⟩ * dhistoryLoad algebra c s f ⟨e,i.2.1,()⟩)*
    keyDecay algebra c s ⟨i.1,()⟩+
    (∑ t : Fin 64,partialScore algebra dot c s scale ⟨i.1,t,()⟩ * valueLoad algebra c s "incoming" f ⟨t,i.2.1,()⟩)=_
  rw [key_decay c s _ hc]
  unfold Body.dvPartial
  simp only [RealScalar.real_cast,RealScalar.real_mul,RealScalar.real_zero]
  rw [RealScalar.chain_sum,RealScalar.chain_sum,zero_add,←Fin.sum_univ_eq_sum_range,←Fin.sum_univ_eq_sum_range]
  apply congrArg₂ (·+·)
  · apply congrArg₂ (·*·)
    · apply Finset.sum_congr rfl
      intro e _
      rw [key_load,dhistory_load]
      rfl
    · rfl
  · apply Finset.sum_congr rfl
    intro t _
    rw [partial_score,incoming_load]
    rfl

#axiomsClean query_body
#axiomsClean key_body
#axiomsClean partial_body
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal.QKV

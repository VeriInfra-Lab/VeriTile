import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelReference

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.FiniteReadLocality
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

/-- Equality only of decoded source reads in a finite region. No storage-tag
or exterior-memory condition is introduced by the finite input adapter. -/
def Reads (r : RegionName) (size : Nat) (a b : Memory α) : Prop :=
  ∀ off, off < size → (a r off).read .real = (b r off).read .real

theorem raw_matrix (M : Algebra α) (a b : Memory α) (r : RegionName)
    (heads rows cols head row col br bc : Nat) (inside : head < heads)
    (same : Reads r (heads*rows*cols) a b) :
    rawLoad M a r (head*(rows*cols)) rows cols cols 1 row col (br:=br) (bc:=bc) =
      rawLoad M b r (head*(rows*cols)) rows cols cols 1 row col := by
  funext i
  unfold rawLoad
  split_ifs with active
  · apply same
    have bound := packed_lt (packed_lt inside active.1) active.2
    convert bound using 1 <;> ring
  · rfl

theorem load_matrix (M : Algebra α) (a b : Memory α) (r : RegionName)
    (heads rows cols head row col br bc : Nat) (inside : head < heads)
    (same : Reads r (heads*rows*cols) a b) :
    load M a r (head*(rows*cols)) rows cols cols 1 row col (br:=br) (bc:=bc) =
      load M b r (head*(rows*cols)) rows cols cols 1 row col := by
  unfold load
  rw [raw_matrix M a b r heads rows cols head row col br bc inside same]

theorem load_transpose (M : Algebra α) (a b : Memory α) (r : RegionName)
    (heads rows cols head row col br bc : Nat) (inside : head < heads)
    (same : Reads r (heads*rows*cols) a b) :
    load M a r (head*(rows*cols)) cols rows 1 cols col row (br:=br) (bc:=bc) =
      load M b r (head*(rows*cols)) cols rows 1 cols col row := by
  funext i
  unfold load rawLoad
  split_ifs with active
  · apply congrArg M.fp32Load
    apply same
    have bound := packed_lt (packed_lt inside active.2) active.1
    convert bound using 1 <;> ring
  · rfl

structure StateReads (p : Parameters) (rev seed : Bool) (a b : Memory α) : Prop where
  left : Reads (if rev then "query" else "key") (p.B*p.H*p.T*p.K) a b
  right : Reads (if rev then "incoming" else "value") (p.B*p.H*p.T*p.V) a b
  initial : seed=Bool.true → Reads (seedName rev) (p.B*p.H*p.K*p.V) a b

theorem state_initial (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (rev seed : Bool) (head : pid.z < p.B*p.H) (same : StateReads p rev seed a b) :
    stateInitial M p pid a rev seed = stateInitial M p pid b rev seed := by
  cases hs : seed
  · unfold stateInitial
    simp only [hs,Bool.false_eq_true,ite_false]
  · have equal := raw_matrix M a b (seedName rev) (p.B*p.H) p.K p.V pid.z
      (pid.x*p.BK) (pid.y*p.BV) p.BK p.BV head (same.initial hs)
    have normalized := congrArg (fun f => fun i => cast M (f i)) equal
    unfold stateInitial
    simpa only [ite_true,Nat.mul_assoc] using normalized

theorem state_left (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (scale : ℝ) (rev seed : Bool) (head : pid.z < p.B*p.H)
    (same : StateReads p rev seed a b) (n : Nat) :
    stateLeft M p pid a scale rev n = stateLeft M p pid b scale rev n := by
  have equal := load_transpose M a b (if rev then "query" else "key")
    (p.B*p.H) p.T p.K pid.z (chunk p rev n*p.BT) (pid.x*p.BK) p.BK p.BT head same.left
  unfold stateLeft
  cases rev <;> simp only [Bool.false_eq_true,ite_false,ite_true] at equal ⊢ <;> rw [equal]

theorem state_right (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (rev seed : Bool) (head : pid.z < p.B*p.H) (same : StateReads p rev seed a b) (n : Nat) :
    stateRight M p pid a rev n = stateRight M p pid b rev n := by
  have equal := load_matrix M a b (if rev then "incoming" else "value")
    (p.B*p.H) p.T p.V pid.z (chunk p rev n*p.BT) (pid.y*p.BV) p.BT p.BV head same.right
  unfold stateRight
  cases rev <;> simp only [Bool.false_eq_true,ite_false,ite_true] at equal ⊢ <;> rw [equal]

theorem state_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (rev seed : Bool) (head : pid.z < p.B*p.H)
    (same : StateReads p rev seed a b) (n : Nat) :
    stateValues M D p pid a scale rev seed n = stateValues M D p pid b scale rev seed n := by
  induction n with
  | zero => exact state_initial M p pid a b rev seed head same
  | succ n ih =>
      simp only [stateValues,ih]
      unfold stateNext
      simp only [state_left M p pid a b scale rev seed head same,
        state_right M p pid a b rev seed head same]

theorem query_tile (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (head : pid.z < p.B*p.H) (same : Reads "query" (p.B*p.H*p.T*p.K) a b) (kb : Nat) :
    queryTile M p pid a kb = queryTile M p pid b kb :=
  load_matrix M a b "query" (p.B*p.H) p.T p.K pid.z (pid.y*p.BT) (kb*p.BK) p.BT p.BK head same

theorem key_tile (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (head : pid.z < p.B*p.H) (same : Reads "key" (p.B*p.H*p.T*p.K) a b) (kb : Nat) :
    keyTile M p pid a kb = keyTile M p pid b kb :=
  load_transpose M a b "key" (p.B*p.H) p.T p.K pid.z (pid.y*p.BT) (kb*p.BK) p.BK p.BT head same

theorem value_tile (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α) (r : RegionName)
    (head : pid.z < p.B*p.H) (same : Reads r (p.B*p.H*p.T*p.V) a b) (vb : Nat) :
    valueTile M p pid a r vb = valueTile M p pid b r vb :=
  load_matrix M a b r (p.B*p.H) p.T p.V pid.z (pid.y*p.BT) (vb*p.BV) p.BT p.BV head same

theorem forward_history (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (head : pid.z < p.B*p.H) (time : pid.y < p.NT)
    (same : Reads "history" (p.B*p.H*p.NT*p.K*p.V) a b) (kb : Nat) :
    forwardHistoryTile M p pid a kb = forwardHistoryTile M p pid b kb := by
  have equal := load_matrix M a b "history" (p.B*p.H*p.NT) p.K p.V (pid.z*p.NT+pid.y)
    (kb*p.BK) (pid.x*p.BV) p.BK p.BV (packed_lt head time) same
  simpa only [forwardHistoryTile,Nat.add_mul,Nat.mul_assoc] using equal

structure OutputReads (p : Parameters) (a b : Memory α) : Prop where
  query : Reads "query" (p.B*p.H*p.T*p.K) a b
  key : Reads "key" (p.B*p.H*p.T*p.K) a b
  value : Reads "value" (p.B*p.H*p.T*p.V) a b
  history : Reads "history" (p.B*p.H*p.NT*p.K*p.V) a b

theorem output_states (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (head : pid.z < p.B*p.H) (time : pid.y < p.NT)
    (same : OutputReads p a b) (n : Nat) :
    outputStates M D p pid a n = outputStates M D p pid b n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [outputStates,outputNext,ih,query_tile M p pid a b head same.query,
        key_tile M p pid a b head same.key,forward_history M p pid a b head time same.history]

theorem output_result (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (head : pid.z < p.B*p.H) (time : pid.y < p.NT)
    (same : OutputReads p a b) :
    outputResult M D p pid a scale (outputStates M D p pid a (keyBlocks p)) =
      outputResult M D p pid b scale (outputStates M D p pid b (keyBlocks p)) := by
  unfold outputResult
  simp only [output_states M D p pid a b head time same,
    value_tile M p pid a b "value" head same.value]

structure GradientReads (p : Parameters) (a b : Memory α) : Prop where
  query : Reads "query" (p.B*p.H*p.T*p.K) a b
  key : Reads "key" (p.B*p.H*p.T*p.K) a b
  value : Reads "value" (p.B*p.H*p.T*p.V) a b
  incoming : Reads "incoming" (p.B*p.H*p.T*p.V) a b
  history : Reads "history" (p.B*p.H*p.NT*p.K*p.V) a b
  historyGradient : Reads "historyGradient" (p.B*p.H*p.NT*p.K*p.V) a b

theorem grad_query (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (head : pid.z < p.B*p.H) (same : Reads "query" (p.B*p.H*p.T*p.K) a b) :
    gradQuery M p pid a = gradQuery M p pid b :=
  load_transpose M a b "query" (p.B*p.H) p.T p.K pid.z (pid.y*p.BT) (pid.x*p.BK) p.BK p.BT head same

theorem grad_key (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (head : pid.z < p.B*p.H) (same : Reads "key" (p.B*p.H*p.T*p.K) a b) :
    gradKey M p pid a = gradKey M p pid b :=
  load_matrix M a b "key" (p.B*p.H) p.T p.K pid.z (pid.y*p.BT) (pid.x*p.BK) p.BT p.BK head same

/-- Retains the original combined NT*K mask, including padded key lanes
which read the following history chunk. -/
theorem grad_history (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (head : pid.z < p.B*p.H) (same : Reads "history" (p.B*p.H*p.NT*p.K*p.V) a b) (vb : Nat) :
    gradHistory M p pid a vb = gradHistory M p pid b vb := by
  have equal := load_transpose M a b "history" (p.B*p.H) (p.NT*p.K) p.V pid.z
    (pid.y*p.K+pid.x*p.BK) (vb*p.BV) p.BV p.BK head
    (by simpa only [Nat.mul_assoc] using same)
  simpa only [gradHistory,Nat.mul_assoc] using equal

theorem grad_history_gradient (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (head : pid.z < p.B*p.H) (same : Reads "historyGradient" (p.B*p.H*p.NT*p.K*p.V) a b) (vb : Nat) :
    gradHistoryGradient M p pid a vb = gradHistoryGradient M p pid b vb := by
  have equal := load_matrix M a b "historyGradient" (p.B*p.H) (p.NT*p.K) p.V pid.z
    (pid.y*p.K+pid.x*p.BK) (vb*p.BV) p.BK p.BV head
    (by simpa only [Nat.mul_assoc] using same)
  simpa only [gradHistoryGradient,Nat.mul_assoc] using equal

theorem gradient_states (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (head : pid.z < p.B*p.H)
    (same : GradientReads p a b) (n : Nat) :
    gradStates M D p pid a scale n = gradStates M D p pid b scale n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [gradStates,ih,gradNext,value_tile M p pid a b "incoming" head same.incoming,
        value_tile M p pid a b "value" head same.value,grad_history M p pid a b head same.history,
        grad_history_gradient M p pid a b head same.historyGradient]

theorem gradient_value (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (head : pid.z < p.B*p.H)
    (same : GradientReads p a b) (vb : Nat) :
    gradValue M D p pid a scale vb = gradValue M D p pid b scale vb := by
  unfold gradValue gradScore
  rw [grad_query M p pid a b head same.query,grad_key M p pid a b head same.key,
    value_tile M p pid a b "incoming" head same.incoming,
    grad_history_gradient M p pid a b head same.historyGradient]

theorem gradient_results (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (head : pid.z < p.B*p.H) (same : GradientReads p a b) :
    gradQueryResult M D p pid a scale (gradStates M D p pid a scale (valueBlocks p)) =
      gradQueryResult M D p pid b scale (gradStates M D p pid b scale (valueBlocks p)) ∧
    gradKeyResult M D p pid a scale (gradStates M D p pid a scale (valueBlocks p)) =
      gradKeyResult M D p pid b scale (gradStates M D p pid b scale (valueBlocks p)) := by
  unfold gradQueryResult gradKeyResult
  simp only [gradient_states M D p pid a b scale head same,grad_query M p pid a b head same.query,
    grad_key M p pid a b head same.key,and_self]

#axiomsClean raw_matrix
#axiomsClean load_transpose
#axiomsClean state_values
#axiomsClean output_result
#axiomsClean gradient_states
#axiomsClean gradient_value
#axiomsClean gradient_results
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.FiniteReadLocality

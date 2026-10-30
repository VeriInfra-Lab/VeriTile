import bench.tritonbench_g.chunk_retention.Sources.KernelNativeModel
import bench.tritonbench_g.chunk_retention.Sources.ForwardOResult

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelStaging ForwardOSource
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

def xs (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α)) :=
  inputs ops literal c s.mem Bool.false Bool.false
def history (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α)) (e j : Nat) : α :=
  lower ops literal ((s.mem "h"
    (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+e*c.V+j)).read .real)

theorem query_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (n : Nat) (i : TileIndex [64,c.BK]) :
    lower ops literal (featureLoad (algebra ops.zero) c s "q" n i)=
      qAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (feature c n i.2.1.val) := by
  simp [featureLoad,qAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero]
theorem key_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (n : Nat) (i : TileIndex [c.BK,64]) :
    lower ops literal (keyLoad (algebra ops.zero) c s n i)=
      kAt ops c (xs ops literal c s) (s.pids 2) (row s i.2.1.val) (feature c n i.1.val) := by
  simp [keyLoad,kAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem history_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (n : Nat) (i : TileIndex [c.BK,c.BV]) :
    lower ops literal (historyLoad (algebra ops.zero) c s n i)=
      masked ops c.K c.V (history ops literal c s) (feature c n i.1.val) (column c s i.2.1.val) := by
  simp [historyLoad,history,masked,algebra,lower_ite,lower,zero,laws.zero]
theorem query_gate (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (n : Nat) (i : TileIndex [64,c.BK]) :
    lower ops literal (queryGate (algebra ops.zero) c s n i)=
      ops.cast (ops.mul
        (qAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (feature c n i.2.1.val))
        (decay ops c (s.pids 2) (i.1.val+1))) := by
  change ops.cast (ops.mul (lower ops literal (featureLoad (algebra ops.zero) c s "q" n i))
    (lower ops literal (KernelModel.decay (algebra ops.zero) c (s.pids 2) ((i.1.val:Int)+1))))=_
  rw [query_load ops literal laws,KernelNative.decay_eq ops literal laws]

def states (ops : Ops α) (c : Shape) (s : State (Term α)) :=
  KernelFold.sequenceValue (ForwardOSource.update (algebra ops.zero) dot c s)
    (⟨fun _ => zero (algebra ops.zero),fun _ => zero (algebra ops.zero)⟩ : Accum c (Term α))
def carryReference (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α))
    (n : Nat) (i : Index c) : α :=
  blockChain ops c.BK
    (fun e => ops.cast (ops.mul (qAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) e)
      (decay ops c (s.pids 2) (i.1.val+1))))
    (fun e => masked ops c.K c.V (history ops literal c s) e (column c s i.2.1.val)) n ops.zero
def scoreReference (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α))
    (n : Nat) (i : ScoreIndex) : α :=
  blockChain ops c.BK
    (fun e => qAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) e)
    (fun e => kAt ops c (xs ops literal c s) (s.pids 2) (row s i.2.1.val) e) n ops.zero

theorem states_carry (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (n : Nat) (i : Index c) :
    lower ops literal ((states ops c s n).1 i)=carryReference ops literal c s n i := by
  induction n with
  | zero => exact laws.zero
  | succ n ih =>
    change lower ops literal (.binary .add ((states ops c s n).1 i)
      (.pendingDot c.BK (fun e => queryGate (algebra ops.zero) c s n ⟨i.1,e,()⟩)
        (fun e => historyLoad (algebra ops.zero) c s n ⟨e,i.2.1,()⟩)))=_
    rw [consume_dot,ih]
    apply chain_congr
    · intro e he
      simp only [he,↓reduceDIte]
      exact query_gate ops literal laws c s n ⟨i.1,⟨e,he⟩,()⟩
    · intro e he
      simp only [he,↓reduceDIte]
      exact history_load ops literal laws c s n ⟨⟨e,he⟩,i.2.1,()⟩

theorem states_score (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (n : Nat) (i : ScoreIndex) :
    lower ops literal ((states ops c s n).2 i)=scoreReference ops literal c s n i := by
  induction n with
  | zero => exact laws.zero
  | succ n ih =>
    change lower ops literal (.binary .add ((states ops c s n).2 i)
      (.pendingDot c.BK (fun e => featureLoad (algebra ops.zero) c s "q" n ⟨i.1,e,()⟩)
        (fun e => keyLoad (algebra ops.zero) c s n ⟨e,i.2.1,()⟩)))=_
    rw [consume_dot,ih]
    apply chain_congr
    · intro e he
      simp only [he,↓reduceDIte]
      exact query_load ops literal laws c s n ⟨i.1,⟨e,he⟩,()⟩
    · intro e he
      simp only [he,↓reduceDIte]
      exact key_load ops literal laws c s n ⟨⟨e,he⟩,i.2.1,()⟩

theorem accumulated_states (ops : Ops α) (c : Shape) (s : State (Term α)) :
    accumulated (algebra ops.zero) dot c s=states ops c s c.keyBlocks :=
  KernelFold.from_zero _ _ _

theorem attention_eq (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (i : ScoreIndex) :
    lower ops literal (attention (algebra ops.zero) dot c s i)=
      ops.cast (ops.mul (Body.score ops c (xs ops literal c s) (s.pids 2) (s.pids 1) i.1.val i.2.1.val)
        (causalWeight ops c (s.pids 2) i.1.val i.2.1.val)) := by
  have hmul : ∀ n b, lower ops literal (.binary .mul ((states ops c s n).2 i) b)=
      ops.mul (lower ops literal ((states ops c s n).2 i)) (lower ops literal b) := by
    intro n b
    cases n <;> rfl
  change ops.cast (lower ops literal (.binary .mul
    ((accumulated (algebra ops.zero) dot c s).2 i) (scoreDecay (algebra ops.zero) c s i)))=_
  rw [accumulated_states,hmul,states_score ops literal laws]
  have hd : lower ops literal (scoreDecay (algebra ops.zero) c s i)=
      causalWeight ops c (s.pids 2) i.1.val i.2.1.val := by
    simp only [scoreDecay,lower_ite,KernelNative.decay_eq ops literal laws,causalWeight]
    rw [show lower ops literal (zero (algebra ops.zero))=ops.zero from laws.zero]
  rw [hd]
  rfl

theorem value_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (i : Index c) :
    lower ops literal (valueLoad (algebra ops.zero) c s i)=
      vAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (column c s i.2.1.val) := by
  simp [valueLoad,active,outputAddress,vAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero]

theorem value_body (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (scale : ℝ) (i : Index c) :
    lower ops literal (value (algebra ops.zero) dot c scale s i)=
      Body.output ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
        (history ops literal c s) i.1.val (column c s i.2.1.val) := by
  change ops.cast (ops.mul (lower ops literal (.binary .add
    ((accumulated (algebra ops.zero) dot c s).1 i)
    (.pendingDot 64 (fun t => attention (algebra ops.zero) dot c s ⟨i.1,t,()⟩)
      (fun t => valueLoad (algebra ops.zero) c s ⟨t,i.2.1,()⟩)))) (literal scale))=_
  rw [consume_dot,accumulated_states,states_carry ops literal laws]
  congr 2
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact attention_eq ops literal laws c s ⟨i.1,⟨t,ht⟩,()⟩
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact value_load ops literal laws c s ⟨⟨t,ht⟩,i.2.1,()⟩

theorem source_result (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (scale : ℝ) :
    ∃ t, FP.Structural.exec (engine (withDot (algebra ops.zero) dot)) (kernel c scale) s=some t ∧
      (∀ i, active c s i → lower ops literal ((t.mem "o" (outputAddress c s i)).read .real)=
        Body.output ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
          (history ops literal c s) i.1.val (column c s i.2.1.val)) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"o" ∨ ∀ i, active c s i → off≠outputAddress c s i) → t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory (algebra ops.zero) dot c scale s
  refine ⟨t,ht,?_,?_⟩
  · intro i hi
    rw [hm,readback (algebra ops.zero) dot c scale s i hi (address_injective c s)]
    exact value_body ops literal laws c s scale i
  · intro r off hf
    rw [hm]
    exact frame (algebra ops.zero) dot c scale s r off hf

#axiomsClean value_body
#axiomsClean source_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O

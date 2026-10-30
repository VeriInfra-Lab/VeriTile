import bench.repairs.tritonbench_g.apply_penalty.Proofs.Execution
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Projection
import bench.tritonbench_g.apply_penalty.Proofs.Real

namespace VeriTile.Bench.Repairs.ApplyPenalty.RealProof
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

def ops : ScalarOps ℝ where
  zero := 0
  fromInt32 := fun c => (c.toInt : ℝ)
  positive := fun x => decide (0 < x)
  divide := (· / ·)
  multiply := (· * ·)
  subtract := (· - ·)
  subtractProduct := fun s c f => s-c*f

theorem arithmetic_formula (x p f r : ℝ) (c : BitVec 32) :
    (arithmetic ops x p f r c).result=penalty x c.toInt p f r := by
  simp [arithmetic,ops,penalty]

def programState (a : Arguments) (s : BlockState) (b : Nat) : BlockState :=
  (s.withPids (fun ax => if ax=0 then b else 0)).withNumPids
    (fun ax => if ax=0 then a.batches else 1)

def source (a : Arguments) : Kernel :=
  (realKernel "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum"
    a.vocab 1 a.block).toAlgKernel

def capacities (a : Arguments) : RegionBounds := fun r =>
  if r="logits" then a.capacity 0 else if r="presence" then a.capacity 1
  else if r="frequency" then a.capacity 2 else if r="repetition" then a.capacity 3
  else if r="ids" then a.capacity 4 else if r="counts" then a.capacity 5
  else if r="cumsum" then a.capacity 6 else 0

/-- Entry read binding for immutable roles. These are input observations,
not execution/output assumptions. Natural channels decode admitted int32. -/
structure ReadOnlyBinding (a : Arguments) (input : Inputs ℝ) (s : BlockState) : Prop where
  presence : ∀ b,s.readMem "presence" b=input.presence b
  frequency : ∀ b,s.readMem "frequency" b=input.frequency b
  repetition : ∀ b,s.readMem "repetition" b=input.repetition b
  ids : ∀ k,s.readMemValue .nat "ids" k=a.token k
  counts : ∀ k,s.readMemValue .nat "counts" k=a.count k
  cumsum : ∀ b,s.readMemValue .nat "cumsum" b=a.start b

/-- All scalar loads and active metadata/logit loads and stores are bounded
for this actual source AST. Bounds are derived from the wrapper contract. -/
theorem source_traceSafe (a : Arguments) (h : InputContract a) (input : Inputs ℝ)
    (s : BlockState) (binding : ReadOnlyBinding a input s) (b : Nat) (hb : b<a.batches) :
    Kernel.TraceSafe (capacities a) (source a) (programState a s b) := by
  have p := h.penalties 1 (by decide) (by decide)
  have f := h.penalties 2 (by decide) (by decide)
  have r := h.penalties 3 (by decide) (by decide)
  have c := h.cumsum (by omega)
  have windows := (h.windows b hb).2.2.1
  apply Projection.safe_pinned Bool.false
    "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum"
    a.vocab 1 a.block (capacities a) (programState a s b) (a.start b) (a.stop b)
    (fun j => a.token (a.start b+j.val))
    (binding.cumsum b) (binding.cumsum (b+1)) (fun _ _ => binding.ids _)
  · simpa [programState,capacities] using (lt_of_lt_of_le hb p)
  · simpa [programState,capacities] using (lt_of_lt_of_le hb f)
  · simpa [programState,capacities] using (lt_of_lt_of_le hb r)
  · simpa [programState,capacities] using (show b<a.capacity 6 by omega)
  · simpa [programState,capacities] using (show b+1<a.capacity 6 by omega)
  · intro j hj; change a.start b+j.val<a.capacity 4; omega
  · intro j hj; change a.start b+j.val<a.capacity 5; omega
  · intro j hj
    simpa [programState,capacities] using offset_bound a h b (a.start b+j.val) hb (by omega) hj

theorem binding_frame (a : Arguments) (input : Inputs ℝ) (s t : BlockState)
    (h : ReadOnlyBinding a input s)
    (frame : ∀ r o,r ≠ "logits" → t.mem r o=s.mem r o) : ReadOnlyBinding a input t := by
  constructor
  · intro b; simpa only [BlockState.readMem,frame "presence" b (by decide)] using h.presence b
  · intro b; simpa only [BlockState.readMem,frame "frequency" b (by decide)] using h.frequency b
  · intro b; simpa only [BlockState.readMem,frame "repetition" b (by decide)] using h.repetition b
  · intro k; simpa only [BlockState.readMemValue,BlockState.readMemTyped,frame "ids" k (by decide)] using h.ids k
  · intro k; simpa only [BlockState.readMemValue,BlockState.readMemTyped,frame "counts" k (by decide)] using h.counts k
  · intro b; simpa only [BlockState.readMemValue,BlockState.readMemTyped,frame "cumsum" b (by decide)] using h.cumsum b

/-- Reuse the complete existing AST proof, then identify its stored values and
frame with the operational direct row interpreter. -/
theorem program_run (a : Arguments) (h : InputContract a) (input : Inputs ℝ)
    (s : BlockState) (binding : ReadOnlyBinding a input s) (b : Nat) (hb : b<a.batches) :
    ∃ t,exec (source a) (programState a s b)=some t ∧
      (∀ q,t.readMem "logits" q=rowRun ops a input (s.readMem "logits") b q) ∧
      (∀ r o,r ≠ "logits" → t.mem r o=s.mem r o) := by
  let ids : Fin a.block → Nat := fun j => a.token (a.start b+j.val)
  let counts : Fin a.block → Nat := fun j => a.count (a.start b+j.val)
  let xs : Fin a.block → ℝ := fun j => s.readMem "logits" (address a b j.val)
  obtain ⟨t,he,hv,hf⟩ := Projection.region_run Bool.false
    "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum"
    a.vocab 1 a.block (programState a s b) (input.presence b) (input.frequency b)
    (input.repetition b) (a.start b) (a.stop b) ids counts xs
    (binding.presence b) (binding.frequency b) (binding.repetition b)
    (binding.cumsum b) (binding.cumsum (b+1))
    (by intro j _; exact binding.ids _) (by intro j _; exact binding.counts _)
    (by intro j _; rfl)
  have injective : ∀ j k : Fin a.block,a.start b+j.val<a.stop b → a.start b+k.val<a.stop b →
      b*a.vocab+ids j=b*a.vocab+ids k → j=k := by
    intro j k hj hk same
    apply Fin.ext
    exact (address_unique a h b b j.val k.val hb hb
      ((event_iff _ _ _).mpr ⟨j.isLt,hj⟩) ((event_iff _ _ _).mpr ⟨k.isLt,hk⟩) same).2
  refine ⟨t,he,?_,fun r o hr => hf r o (Or.inl (by simpa using hr))⟩
  intro q
  by_cases selected : ∃ lane,lane ∈ rowEvents a b ∧ q=address a b lane
  · obtain ⟨lane,hl,rfl⟩ := selected
    have ev := (event_iff _ _ _).mp hl
    let j : Fin a.block := ⟨lane,ev.1⟩
    have result := hv injective j ev.2
    rw [row_selected ops a h input (s.readMem "logits") b lane hb hl]
    have nonneg := ((h.windows b hb).2.2.2.2.1 _ (by omega) ev.2).2.2
    have countEq : ((a.count (a.start b+lane) : Nat) : ℝ) =
        ((a.counts (a.start b+lane)).toInt : ℝ) := by
      rw [Arguments.count,← Int.cast_natCast,Int.toNat_of_nonneg nonneg]
    simpa only [VeriTile.Bench.TritonBenchG.ApplyPenalty.penaltyValuePure,
      TiledOptimizer.lionPenalty,programState,BlockState.withPids,BlockState.withNumPids,
      ids,counts,xs,j,address,value,arithmetic,ops,countEq,decide_eq_true_eq,
      ite_eq_iff] using result
  · have miss : ∀ lane ∈ rowEvents a b,q ≠ address a b lane := by
      intro lane hl eq; exact selected ⟨lane,hl,eq⟩
    rw [row_frame ops a input (s.readMem "logits") b q miss]
    have frame := hf "logits" q (Or.inr (by
      intro j hj
      exact miss j.val ((event_iff _ _ _).mpr ⟨j.isLt,hj⟩)))
    simpa only [BlockState.readMem,frame] using
      (show (programState a s b).readMem "logits" q=s.readMem "logits" q from rfl)

def runSource (a : Arguments) : List Nat → BlockState → Option BlockState
  | [],s => some s
  | b::rest,s => do
      let t ← exec (source a) (programState a s b)
      runSource a rest t

theorem whole_source (a : Arguments) (h : InputContract a) (input : Inputs ℝ)
    (order : List Nat) (bounds : ∀ b ∈ order,b<a.batches) (s : BlockState)
    (binding : ReadOnlyBinding a input s) :
    ∃ t,runSource a order s=some t ∧
      (∀ q,t.readMem "logits" q=runRows ops a input order (s.readMem "logits") q) ∧
      (∀ r o,r ≠ "logits" → t.mem r o=s.mem r o) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,by intros; rfl,by intros; rfl⟩
  | cons b rest ih =>
    obtain ⟨mid,hm,hv,hf⟩ := program_run a h input s binding b (bounds b (by simp))
    obtain ⟨t,ht,tv,tf⟩ := ih (fun c hc => bounds c (by simp [hc])) mid
      (binding_frame a input s mid binding hf)
    refine ⟨t,by simp only [runSource,hm]; exact ht,?_,?_⟩
    · intro q
      rw [tv,show mid.readMem "logits"=rowRun ops a input (s.readMem "logits") b from funext hv]
      rfl
    · intro r o hr; exact (tf r o hr).trans (hf r o hr)

#axiomsClean arithmetic_formula
#axiomsClean source_traceSafe
#axiomsClean program_run
#axiomsClean whole_source
end
end VeriTile.Bench.Repairs.ApplyPenalty.RealProof

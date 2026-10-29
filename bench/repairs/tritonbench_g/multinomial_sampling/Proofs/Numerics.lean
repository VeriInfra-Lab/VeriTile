import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.NumericTrees
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Native

namespace VeriTile.Bench.Repairs.MultinomialSampling.Numerics
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open NumericTrees
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000
abbrev Q (plans : Plans) := software .fp32 plans
abbrev R := Real.primitives

structure Inputs (p : Parameters) (word : State Word) (real : State ℝ) : Prop where
  pids : word.pids = real.pids
  undef : word.undef .int = real.undef .int
  integers : ∀ r o, (word.mem r o).read .int = (real.mem r o).read .int
  score : ∀ o, decode ((word.mem p.scores o).read .real) = (real.mem p.scores o).read .real

theorem zero_decode : decode (0 : Word) = 0 := by
  have h : (f32 (0 : Word)).toRat? = some 0 := by decide +kernel
  unfold decode
  rw [h]
  simp

theorem f32_put (v : FP.Value .fp32) : f32 (put32 v) = v := by
  cases v with
  | mk b =>
    apply congrArg FP.Value.mk
    change (b.setWidth 64).setWidth 32 = b
    rw [BitVec.setWidth_setWidth_of_le b (by decide : 32 ≤ 64)]
    exact BitVec.setWidth_eq b

theorem cast_decode (x : Word) : decode (put32 (f32 x)) = decode x := by
  unfold decode
  rw [f32_put]

theorem row_eq (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (i : Row p) : row p word i = row p real i := by
  simp [row,h.pids]

theorem int_load_eq (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (r : RegionName) (o : Nat) (b : Bool) :
    intLoad word r o b = intLoad real r o b := by
  simp [intLoad,h.integers,h.undef]

theorem sample_eq (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (i : Row p) :
    decode (sample (Q plans) p word i) = sample R p real i := by
  have hs : seedValue p word i = seedValue p real i := by
    simp only [seedValue,active,row_eq p word real h]
    exact int_load_eq p word real h _ _ _
  have ho : offsetValue p word i = offsetValue p real i := by
    simp only [offsetValue,active,row_eq p word real h,int_load_eq p word real h]
  rw [Real.primitive_sample]
  change decode (put32 (Random.rand (seedValue p word i) (offsetValue p word i) .bits32 10)) = _
  unfold decode
  rw [f32_put,hs,ho,Random.rand_toRat_randRat]
  rfl

theorem scores_eq (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (b : Nat) (i : Lane p) :
    decode (scores (Q plans) .fp32 p word b i) = scores R .fp32 p real b i := by
  have hr := row_eq p word real h (i.1,PUnit.unit)
  have ho : scoreOffset p word b i = scoreOffset p real b i := by simp [scoreOffset,hr]
  change decode (put32 (f32 (if active p word (i.1,PUnit.unit) ∧ b+i.2.1.val < p.tokens then
    (word.mem p.scores (scoreOffset p word b i)).read .real else 0))) =
    if active p real (i.1,PUnit.unit) ∧ b+i.2.1.val < p.tokens then
      (real.mem p.scores (scoreOffset p real b i)).read .real else 0
  rw [cast_decode]
  simp only [active,hr,ho]
  split
  · exact h.score (scoreOffset p real b i)
  · exact zero_decode

def ExactBinary (op : Word → Word → Word) (realOp : ℝ → ℝ → ℝ) (x y : Word) : Prop :=
  Finite x ∧ Finite y ∧ Finite (op x y) ∧ decode (op x y) = realOp (decode x) (decode y)

def ExactCompare (op : Word → Word → Bool) (realOp : ℝ → ℝ → Bool) (x y : Word) : Prop :=
  Finite x ∧ Finite y ∧ op x y = realOp (decode x) (decode y)

/-- These conditions name only actual operands at local numerical nodes.
Scan/max conditions recursively enumerate binary joins of the selected
software tree. RNG, integer reduction, argmax, labels, host memory and kernel
output are absent from the numerical conditions. -/
structure LocalExact (plans : Plans) (p : Parameters) (s : State Word) (k : Nat) : Prop where
  scans : ∀ i : Lane p, ScanLocal plans p (scores (Q plans) .fp32 p s (k*p.blockTokens)) i
  maxima : ∀ i : Row p, MaxLocal plans (fun j : Fin p.blockTokens =>
    scanned (Q plans) .fp32 p s (k*p.blockTokens) (i.1,j,PUnit.unit))
  accumulated : ∀ i : Row p, ExactBinary (Q plans).add (·+·)
    ((rounds (Q plans) .fp32 p s k).acc i)
    ((Q plans).maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned (Q plans) .fp32 p s (k*p.blockTokens)) i)
  cumulative : ∀ i : Lane p, ExactBinary (Q plans).add (·+·)
    ((rounds (Q plans) .fp32 p s k).acc (i.1,PUnit.unit))
    (scanned (Q plans) .fp32 p s (k*p.blockTokens) i)
  preceding : ∀ i : Lane p, ExactBinary (Q plans).sub (·-·)
    (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.cumulative (Q plans) .fp32 p s (k*p.blockTokens) (rounds (Q plans) .fp32 p s k) i)
    (scores (Q plans) .fp32 p s (k*p.blockTokens) i)
  lower : ∀ i : Lane p, ExactCompare (Q plans).lt (fun x y => decide (x<y))
    ((Q plans).sub (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.cumulative (Q plans) .fp32 p s (k*p.blockTokens) (rounds (Q plans) .fp32 p s k) i)
      (scores (Q plans) .fp32 p s (k*p.blockTokens) i))
    (sample (Q plans) p s (i.1,PUnit.unit))
  upper : ∀ i : Lane p, ExactCompare (Q plans).le (fun x y => decide (x≤y))
    (sample (Q plans) p s (i.1,PUnit.unit))
    (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.cumulative (Q plans) .fp32 p s (k*p.blockTokens) (rounds (Q plans) .fp32 p s k) i)

theorem scanned_eq (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (k : Nat) (nodes : LocalExact plans p word k) (i : Lane p) :
    decode (scanned (Q plans) .fp32 p word (k*p.blockTokens) i) = scanned R .fp32 p real (k*p.blockTokens) i := by
  change decode (forwardScan plans .sum ⟨1,by simp⟩ (scores (Q plans) .fp32 p word (k*p.blockTokens)) i) = _
  rw [scan_decode plans p _ i (nodes.scans i)]
  have he : (fun j => decode (scores (Q plans) .fp32 p word (k*p.blockTokens) j)) =
      scores R .fp32 p real (k*p.blockTokens) := funext (scores_eq plans p word real h _)
  rw [he]
  rfl

theorem maximum_eq (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (k : Nat) (nodes : LocalExact plans p word k) (i : Row p) :
    decode ((Q plans).maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned (Q plans) .fp32 p word (k*p.blockTokens)) i) =
      R.maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned R .fp32 p real (k*p.blockTokens)) i := by
  change decode (maximum plans (fun j : Fin p.blockTokens => scanned (Q plans) .fp32 p word (k*p.blockTokens) (i.1,j,PUnit.unit))) = _
  rw [maximum_decode plans p.tokensDomain.1 _ (nodes.maxima i)]
  have he : (fun j : Fin p.blockTokens => decode (scanned (Q plans) .fp32 p word (k*p.blockTokens) (i.1,j,PUnit.unit))) =
      (fun j : Fin p.blockTokens => scanned R .fp32 p real (k*p.blockTokens) (i.1,j,PUnit.unit)) :=
    funext (fun j => scanned_eq plans p word real h k nodes _)
  rw [he]
  rfl

theorem cumulative_eq (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (k : Nat) (nodes : LocalExact plans p word k)
    (acc : ∀ i, decode ((rounds (Q plans) .fp32 p word k).acc i) = (rounds R .fp32 p real k).acc i)
    (i : Lane p) :
    decode (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.cumulative (Q plans) .fp32 p word (k*p.blockTokens) (rounds (Q plans) .fp32 p word k) i) =
      VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.cumulative R .fp32 p real (k*p.blockTokens) (rounds R .fp32 p real k) i := by
  rw [VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.cumulative,(nodes.cumulative i).2.2.2,acc,scanned_eq plans p word real h k nodes]
  rfl

theorem valid_eq (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (k : Nat) (nodes : LocalExact plans p word k)
    (acc : ∀ i, decode ((rounds (Q plans) .fp32 p word k).acc i) = (rounds R .fp32 p real k).acc i) :
    valid (Q plans) .fp32 p word (k*p.blockTokens) (rounds (Q plans) .fp32 p word k) =
      valid R .fp32 p real (k*p.blockTokens) (rounds R .fp32 p real k) := by
  funext i
  rw [valid,(nodes.lower i).2.2,(nodes.upper i).2.2,(nodes.preceding i).2.2.2,
    cumulative_eq plans p word real h k nodes acc,scores_eq plans p word real h,sample_eq plans p word real h]
  rfl

theorem rounds_eq (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (k : Nat) (nodes : ∀ j, j < k → LocalExact plans p word j) :
    (∀ i, decode ((rounds (Q plans) .fp32 p word k).acc i) = (rounds R .fp32 p real k).acc i) ∧
    (rounds (Q plans) .fp32 p word k).output = (rounds R .fp32 p real k).output := by
  induction k with
  | zero =>
    constructor
    · intro i; exact zero_decode
    · funext i
      change initialOutput p word i = initialOutput p real i
      simp only [initialOutput,active,row_eq p word real h]
      exact int_load_eq p word real h _ _ _
  | succ k ih =>
    obtain ⟨ha,ho⟩ := ih (fun j hj => nodes j (by omega))
    have loc := nodes k (by omega)
    have hv := valid_eq plans p word real h k loc ha
    have hf : found (Q plans) .fp32 p word (k*p.blockTokens) (rounds (Q plans) .fp32 p word k) =
        found R .fp32 p real (k*p.blockTokens) (rounds R .fp32 p real k) := by
      funext i
      unfold found
      rw [hv]
    have hs : selected (Q plans) .fp32 p word (k*p.blockTokens) (rounds (Q plans) .fp32 p word k) =
        selected R .fp32 p real (k*p.blockTokens) (rounds R .fp32 p real k) := by
      funext i
      unfold selected
      rw [hv]
    constructor
    · intro i
      change decode ((Q plans).add ((rounds (Q plans) .fp32 p word k).acc i)
        ((Q plans).maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned (Q plans) .fp32 p word (k*p.blockTokens)) i)) = _
      rw [(loc.accumulated i).2.2.2,ha,maximum_eq plans p word real h k loc]
      rfl
    · funext i
      simp only [rounds,advance,hf,selectedIndex,hs,active,row_eq p word real h,h.integers,ho]
      rfl

theorem input_labels_eq (p : Parameters) (a : Host.Arguments) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (o j : Nat) : Host.inputLabels p a word o j = Host.inputLabels p a real o j := by
  simp only [Host.inputLabels,h.integers]

theorem prepared_integer (p : Parameters) (a : Host.Arguments) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (r : RegionName) (o : Nat) :
    ((Host.prepared p a word).mem r o).read .int = ((Host.prepared p a real).mem r o).read .int := by
  have hi (r : RegionName) (o : Nat) :
      ((VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices p a.toArguments word).mem r o).read .int =
      ((VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices p a.toArguments real).mem r o).read .int := by
    unfold VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices
    split
    · dsimp only
      split
      · rfl
      · exact h.integers r o
    · exact h.integers r o
  unfold Host.prepared VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.kernelInitial VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.cloneFirstColumn
  dsimp only
  split
  · simp only [Cell.read_same,hi]
  · exact hi r o

theorem prepared_inputs (p : Parameters) (a : Host.Arguments) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (fresh : Host.Fresh p a) :
    Inputs (Host.params p a) (Host.prepared p a word) (Host.prepared p a real) := by
  refine ⟨?_,?_,prepared_integer p a word real h,?_⟩
  · change (Host.prepared p a word).pids = (Host.prepared p a real).pids
    unfold Host.prepared VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.kernelInitial VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.cloneFirstColumn VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices
    split <;> exact h.pids
  · change (Host.prepared p a word).undef .int = (Host.prepared p a real).undef .int
    unfold Host.prepared VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.kernelInitial VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.cloneFirstColumn VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices
    split <;> exact h.undef
  · intro o
    change decode (((Host.prepared p a word).mem p.scores o).read .real) =
      ((Host.prepared p a real).mem p.scores o).read .real
    rw [Host.prepared_input p a word p.scores fresh.1.1 (fun h => (fresh.2 h).1),
      Host.prepared_input p a real p.scores fresh.1.1 (fun h => (fresh.2 h).1)]
    exact h.score o

theorem program_inputs (p : Parameters) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (pid : Nat) :
    Inputs p (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState p word word pid) (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState p real real pid) :=
  ⟨rfl,h.undef,h.integers,h.score⟩

def LaunchExact (plans : Plans) (p : Parameters) (a : Host.Arguments) (word : State Word) : Prop :=
  ∀ pid, pid < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid (Host.params p a) → ∀ k, k < count (Host.params p a) →
    LocalExact plans (Host.params p a)
      (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState (Host.params p a) (Host.prepared p a word) (Host.prepared p a word) pid) k

theorem value_eq (plans : Plans) (p : Parameters) (a : Host.Arguments) (word : State Word) (real : State ℝ)
    (h : Inputs p word real) (fresh : Host.Fresh p a) (nodes : LaunchExact plans p a word)
    (o : Nat) (ho : o < p.batches) :
    VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value (Q plans) .fp32 (Host.params p a) (Host.prepared p a word) o =
      VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value R .fp32 (Host.params p a) (Host.prepared p a real) o := by
  let q := Host.params p a
  have hp : o/q.block < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid q := by
    apply (Nat.div_lt_iff_lt_mul q.blockDomain.1).mpr
    have hc := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid_covers q
    change q.batches ≤ _ at hc
    have hb : o < q.batches := ho
    omega
  have relation := program_inputs q (Host.prepared p a word) (Host.prepared p a real)
    (prepared_inputs p a word real h fresh) (o/q.block)
  have eq := (rounds_eq plans q _ _ relation (count q) (nodes (o/q.block) hp)).2
  exact congrFun eq (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.lane q o)

/-- The data-dependent label load is inside the true token dimension. The
only conditional facts are the enumerated local FP arithmetic/comparison
ports; no output or whole execution equality is assumed. -/
theorem selected_in_bounds (plans : Plans) (p : Parameters) (word : State Word) (real : State ℝ)
    (inputs : Inputs p word real) (k : Nat)
    (nodes : ∀ j, j ≤ k → LocalExact plans p word j) (i : Row p)
    (activeWord : active p word i) (nonnegative : CDFBridge.Nonnegative p real i)
    (hit : found (Q plans) .fp32 p word (k*p.blockTokens)
      (rounds (Q plans) .fp32 p word k) i = Bool.true) :
    selected (Q plans) .fp32 p word (k*p.blockTokens)
      (rounds (Q plans) .fp32 p word k) i < p.tokens := by
  have acc := (rounds_eq plans p word real inputs k (fun j hj => nodes j (by omega))).1
  have eq := valid_eq plans p word real inputs k (nodes k le_rfl) acc
  have ha : active p real i := by
    unfold active at activeWord ⊢
    rwa [row_eq p word real inputs] at activeWord
  have hf : found R .fp32 p real (k*p.blockTokens) (rounds R .fp32 p real k) i = Bool.true := by
    unfold found at hit ⊢
    rwa [eq] at hit
  have hs : selected (Q plans) .fp32 p word (k*p.blockTokens) (rounds (Q plans) .fp32 p word k) i =
      selected R .fp32 p real (k*p.blockTokens) (rounds R .fp32 p real k) i := by
    unfold selected
    rw [eq]
  obtain ⟨j,hj⟩ := (CDFBridge.found_iff p real i ha nonnegative k).mp hf
  rw [hs,CDFBridge.selected_chosen p real i ha nonnegative _ hj k hf]
  exact CDFBridge.interval_inside p real i _ _ hj

#axiomsClean selected_in_bounds
#axiomsClean zero_decode
#axiomsClean sample_eq
#axiomsClean scanned_eq
#axiomsClean maximum_eq
#axiomsClean rounds_eq
#axiomsClean prepared_inputs
#axiomsClean value_eq
#guard_msgs (drop info) in
#specNonCircular LocalExact avoiding [Host.invoke,Physical.run,Native.entry]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Numerics

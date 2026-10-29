import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.CDFBridge
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPWrapper

namespace VeriTile.Bench.Repairs.MultinomialSampling.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section
set_option maxHeartbeats 3000000

structure Arguments extends VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.Arguments where
  seedsContiguous : Bool
  offsetsContiguous : Bool

abbrev params (p : Parameters) (a : Arguments) := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.wrapperParameters p a.toArguments
abbrev prepared {α : Type} [Inhabited α] (p : Parameters) (a : Arguments) (s : State α) := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.kernelInitial p a.toArguments s

def invoke {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (p : Parameters) (a : Arguments) (s : State α) : Option (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.Result α) :=
  if a.scoreRank ≠ 2 then none
  else if a.indexShape.any (fun sh => sh != [p.batches,p.tokens]) then none
  else if p.tokens = 1 then some ⟨prepared p a s,[p.batches,1]⟩
  else if !a.seedsContiguous || !a.offsetsContiguous then none
  else if p.tokens = 0 then none
  else (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.run rest P .fp32 (params p a) (prepared p a s)).map fun final => ⟨final,[p.batches]⟩

def Valid (p : Parameters) (a : Arguments) : Prop :=
  a.scoreRank = 2 ∧ a.indexShape.any (fun sh => sh != [p.batches,p.tokens]) = Bool.false ∧
  0 < p.tokens ∧ (p.tokens ≠ 1 → a.seedsContiguous = Bool.true ∧ a.offsetsContiguous = Bool.true)

def expectedMemory {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (a : Arguments) (s : State α) :=
  if p.tokens = 1 then (prepared p a s).mem else VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.expectedMemory P .fp32 (params p a) (prepared p a s)

theorem wrapper_correct {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (p : Parameters) (a : Arguments) (s : State α) (valid : Valid p a)
    (fresh : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.FreshOutput p)
    (seeds : p.tokens ≠ 1 → ∀ k, k < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid (params p a) →
      SeedDomain (params p a) (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState (params p a) (prepared p a s) (prepared p a s) k)) :
    ∃ result, invoke rest P p a s = some result ∧ result.state.mem = expectedMemory P p a s ∧
      result.shape = if p.tokens = 1 then [p.batches,1] else [p.batches] := by
  by_cases token : p.tokens = 1
  · have index := valid.2.1
    simp only [token] at index
    refine ⟨⟨prepared p a s,[p.batches,1]⟩,?_,?_,?_⟩ <;>
      simp [invoke,expectedMemory,valid.1,token,index]
  · have hf : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.FreshOutput (params p a) := fresh
    obtain ⟨final,he,hm⟩ := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.launch_correct rest P .fp32 (params p a) (prepared p a s) hf (seeds token)
    refine ⟨⟨final,[p.batches]⟩,?_,?_,?_⟩
    · simp [invoke,valid.1,valid.2.1,token,(valid.2.2.2 token).1,(valid.2.2.2 token).2,
        Nat.ne_of_gt valid.2.2.1,he]
    · simpa only [expectedMemory,token,↓reduceIte] using hm
    · simp [token]

/-- Input-only host data, in the source's ordinary contiguous Tensor domain.
Default labels are the token ordinal; supplied labels use their actual stride. -/
def inputWeights (p : Parameters) (s : State ℝ) (o j : Nat) : ℝ :=
  if j < p.tokens then (s.mem p.scores (o*p.strideScoreBatch+j*p.strideScoreToken)).read .real else 0

def inputLabels (p : Parameters) (a : Arguments) (s : State α) [Inhabited α] (o j : Nat) : Int :=
  if a.indexShape.isNone then Int.ofNat j
  else (s.mem p.indices (o*p.strideIndexBatch+j*p.strideIndexToken)).read .int

def Result (p : Parameters) (a : Arguments) (s : State ℝ) (o : Nat) (output : Int) : Prop :=
  Spec.Result p.tokens (inputWeights p s o) (inputLabels p a s o)
    ((s.mem p.seeds o).read .int) ((s.mem p.offsets o).read .int) output

/-- Ordinary new-storage contracts: the output clone is fresh; when arange
is needed its new backing storage differs from all caller input regions. -/
def Fresh (p : Parameters) (a : Arguments) : Prop :=
  VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.FreshOutput p ∧
  (a.indexShape.isNone = Bool.true → p.scores ≠ p.indices ∧ p.seeds ≠ p.indices ∧ p.offsets ≠ p.indices)

theorem prepared_input {α : Type} [Inhabited α] (p : Parameters) (a : Arguments) (s : State α)
    (r : RegionName) (hout : r ≠ p.output)
    (hindex : a.indexShape.isNone = Bool.true → r ≠ p.indices) :
    (prepared p a s).mem r = s.mem r := by
  funext o
  apply VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.initialized_frame
  · exact Or.inl hout
  · by_cases h : a.indexShape.isNone = Bool.true
    · exact Or.inr (Or.inl (hindex h))
    · exact Or.inl (Bool.eq_false_iff.mpr h)

theorem prepared_labels {α : Type} [Inhabited α] (p : Parameters) (a : Arguments) (s : State α)
    (fresh : Fresh p a) (o j : Nat) (ho : o < p.batches) (hj : j < p.tokens) :
    ((prepared p a s).mem p.indices (o*(params p a).strideIndexBatch+j*(params p a).strideIndexToken)).read .int =
      inputLabels p a s o j := by
  have hout : p.indices ≠ p.output := fresh.1.2.2.2
  have hframe : (prepared p a s).mem p.indices = (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices p a.toArguments s).mem p.indices := by
    funext z
    simp [prepared,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.kernelInitial,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.cloneFirstColumn,params,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.wrapperParameters,hout]
  rw [hframe]
  cases hi : a.indexShape with
  | none =>
    rw [VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.default_indices p a.toArguments s (by exact hi) o j ho hj]
    simp [inputLabels,hi,Cell.read]
  | some shape =>
    simp [VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices,params,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.wrapperParameters,inputLabels,hi]

theorem global_row (p : Parameters) (s : State α) (o : Nat) :
    row p (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState p s s (o/p.block)) (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.lane p o) = o := by
  dsimp [row,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.lane]
  rw [Nat.mul_comm]
  exact Nat.div_add_mod o p.block

theorem value_result (p : Parameters) (a : Arguments) (s : State ℝ)
    (fresh : Fresh p a) (o : Nat) (ho : o < p.batches)
    (domain : Spec.Domain p.tokens (inputWeights p s o)) :
    Result p a s o (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value Real.primitives .fp32 (params p a) (prepared p a s) o) := by
  let q := params p a
  let t := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState q (prepared p a s) (prepared p a s) (o/q.block)
  let i := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.lane q o
  have hr : row q t i = o := global_row q (prepared p a s) o
  have ha : active q t i := by change row q t i < p.batches; rwa [hr]
  have hs := prepared_input p a s p.scores fresh.1.1 (fun h => (fresh.2 h).1)
  have hse := prepared_input p a s p.seeds fresh.1.2.1 (fun h => (fresh.2 h).2.1)
  have hof := prepared_input p a s p.offsets fresh.1.2.2.1 (fun h => (fresh.2 h).2.2)
  have hw : CDFBridge.weights q t i = inputWeights p s o := by
    funext j
    unfold CDFBridge.weights inputWeights
    change (if j < p.tokens then ((prepared p a s).mem p.scores
      (row q t i*p.strideScoreBatch+j*p.strideScoreToken)).read .real else 0) = _
    rw [hr,hs]
  have hu : sample Real.primitives q t i =
      Spec.sample ((s.mem p.seeds o).read .int) ((s.mem p.offsets o).read .int) := by
    rw [Real.primitive_sample]
    unfold Spec.sample seedValue offsetValue intLoad
    simp only [ha,decide_true,↓reduceIte]
    change Random.randReal (((prepared p a s).mem p.seeds (row q t i)).read .int)
      (BitVec.ofInt 32 (((prepared p a s).mem p.offsets (row q t i)).read .int)).toInt .bits32 10 = _
    rw [hr,hse,hof]
  have hd : Spec.Domain q.tokens (CDFBridge.weights q t i) := by rw [hw]; exact domain
  obtain ⟨j,hj,hv⟩ := CDFBridge.rounds_result q t i ha hd
  have hl : CDFBridge.labels q t i j = inputLabels p a s o j := by
    unfold CDFBridge.labels
    change ((prepared p a s).mem p.indices
      (row q t i*(params p a).strideIndexBatch+j*(params p a).strideIndexToken)).read .int = _
    rw [hr]
    exact prepared_labels p a s fresh o j ho hj.1
  refine ⟨j,?_,?_⟩
  · simpa only [hw,hu] using hj
  · exact hv.trans hl

theorem singleton_result (p : Parameters) (a : Arguments) (s : State ℝ)
    (fresh : Fresh p a) (o : Nat) (ho : o < p.batches) (token : p.tokens = 1)
    (domain : Spec.Domain p.tokens (inputWeights p s o)) :
    Result p a s o (((prepared p a s).mem p.output o).read .int) := by
  have hclone := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.clone_readback p a.toArguments s o ho
  rw [hclone]
  simp only [Cell.read_same]
  have hlabel := prepared_labels p a s fresh o 0 ho (by omega)
  have hout : p.indices ≠ p.output := fresh.1.2.2.2
  have hsame : ((prepared p a s).mem p.indices (o*(params p a).strideIndexBatch)).read .int =
      ((VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices p a.toArguments s).mem p.indices (o*(params p a).strideIndexBatch)).read .int := by
    simp [prepared,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.kernelInitial,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.cloneFirstColumn,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.wrapperParameters,hout]
  simp only [Nat.zero_mul,Nat.add_zero] at hlabel
  rw [← hsame,hlabel]
  unfold Result
  rw [token]
  apply Spec.singleton
  simpa only [token] using domain

/-- Complete repaired host result and logical frame, including singleton
cloning, actual grid coverage, and the input-only canonical categorical spec. -/
theorem whole_real (rest : Algebra ℝ) (p : Parameters) (a : Arguments) (s : State ℝ)
    (valid : Valid p a) (fresh : Fresh p a)
    (domain : ∀ o, o < p.batches → Spec.Domain p.tokens (inputWeights p s o))
    (seeds : p.tokens ≠ 1 → ∀ k, k < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid (params p a) →
      SeedDomain (params p a) (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState (params p a) (prepared p a s) (prepared p a s) k)) :
    ∃ result, invoke rest Real.primitives p a s = some result ∧
      result.shape = (if p.tokens = 1 then [p.batches,1] else [p.batches]) ∧
      (∀ o, o < p.batches → Result p a s o ((result.state.mem p.output o).read .int)) ∧
      (∀ r z, (r ≠ p.output ∨ p.batches ≤ z) →
        (a.indexShape.isNone = Bool.false ∨ r ≠ p.indices ∨ p.tokens ≤ z) →
        result.state.mem r z = s.mem r z) := by
  obtain ⟨result,he,hm,hshape⟩ := wrapper_correct rest Real.primitives p a s valid fresh.1 seeds
  refine ⟨result,he,hshape,?_,?_⟩
  · intro o ho
    rw [hm]
    unfold expectedMemory
    split
    · exact singleton_result p a s fresh o ho ‹_› (domain o ho)
    · change Result p a s o ((VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.expectedMemory Real.primitives .fp32 (params p a) (prepared p a s) p.output o).read .int)
      change Result p a s o ((if p.output = p.output ∧ o < p.batches then
        Cell.mk .int (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value Real.primitives .fp32 (params p a) (prepared p a s) o)
        else (prepared p a s).mem p.output o).read .int)
      rw [if_pos ⟨rfl,ho⟩,Cell.read_same]
      exact value_result p a s fresh o ho (domain o ho)
  · intro r z hout hindex
    rw [hm]
    have hprepared := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.initialized_frame p a.toArguments s r z hout hindex
    unfold expectedMemory
    split
    · exact hprepared
    · change (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.expectedMemory Real.primitives .fp32 (params p a) (prepared p a s)) r z = _
      unfold VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.expectedMemory
      rw [if_neg (by
        change ¬ (r = p.output ∧ z < p.batches)
        rintro ⟨hr,hz⟩
        rcases hout with h | h
        · exact h hr
        · omega)]
      exact hprepared

#axiomsClean wrapper_correct
#axiomsClean prepared_input
#axiomsClean prepared_labels
#axiomsClean value_result
#axiomsClean singleton_result
#axiomsClean whole_real
#specNonCircular Result avoiding [invoke,VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Host

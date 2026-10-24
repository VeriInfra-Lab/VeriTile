import bench.tritonbench_g.multinomial_sampling.Proofs.FPHost

namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue.Host
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-! Logical lowering of the Python wrapper's shape assertions, arange/expand,
clone, single-token early return, and fixed 8-by-128 launch. The supplied-index
profile is signed64, as in the admitted kernel. This is not a claim about other
supplied-index dtypes or a proof of PyTorch's allocator implementation. -/

structure Arguments where
  /-- The wrapper first asserts rank two. In that branch, p.batches and
  p.tokens are the two dimensions obtained from scores.size(). -/
  scoreRank : Nat
  /-- None means arange(tokens).expand_as(scores); some preserves the supplied
  tensor's shape and the parameter record's strides. -/
  indexShape : Option (List Nat)

def wrapperParameters (p : Parameters) (a : Arguments) : Parameters :=
  { p with
    strideIndexBatch := if a.indexShape.isNone then
      (if p.batches = 1 then p.tokens else 0) else p.strideIndexBatch
    strideIndexToken := if a.indexShape.isNone then 1 else p.strideIndexToken
    block := 8
    blockTokens := 128
    blockDomain := ⟨by decide, by decide, 3, by decide⟩
    tokensDomain := ⟨by decide, by decide, 7, by decide⟩ }

/-- Allocate the default arange backing storage. Expansion does not copy the
tensor. The leading singleton stride is tokens; actual batch broadcasting
uses stride zero. -/
def prepareIndices (p : Parameters) (a : Arguments) (initial : State α) : State α :=
  if a.indexShape.isNone then
    { initial with mem := fun r o =>
        if r = p.indices ∧ o < p.tokens then .mk .int (Int.ofNat o) else initial.mem r o }
  else initial

/-- indices[:, 0].clone(): output is a separate contiguous vector. -/
def cloneFirstColumn (p : Parameters) (initial : State α) : State α :=
  { initial with mem := fun r o =>
      if r = p.output ∧ o < p.batches then (Cell.mk .int ((initial.mem p.indices (o * p.strideIndexBatch)).read .int))
      else initial.mem r o }

def kernelInitial (p : Parameters) (a : Arguments) (initial : State α) : State α :=
  cloneFirstColumn (wrapperParameters p a) (prepareIndices p a initial)

/-- The early return is a B-by-1 int64 tensor, whereas the kernel branch
returns a vector of length B. No seed, offset, or indices read occurs here. -/
def zeros (p : Parameters) (initial : State α) : State α :=
  { initial with mem := fun r o =>
      if r = p.output ∧ o < p.batches then Cell.mk .int 0 else initial.mem r o }

structure Result (α : Type) where
  state : State α
  shape : List Nat

/-- None represents a failed shape assertion or the out-of-bounds [:,0]
selection for an empty token dimension. Device and storage validity are
external input conditions of the logical memory model. -/
def invoke (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (a : Arguments) (initial : State α) : Option (Result α) :=
  if a.scoreRank ≠ 2 then none
  else if p.tokens = 1 then some ⟨zeros p initial, [p.batches, 1]⟩
  else if a.indexShape.any (fun sh => sh != [p.batches, p.tokens]) then none
  else if p.tokens = 0 then none
  else (run rest P d (wrapperParameters p a) (kernelInitial p a initial)).map
    fun final => ⟨final, [p.batches]⟩

/-- Output reference evaluated on host-prepared initial memory. It uses the
independent token recurrence and never invokes the kernel or the wrapper. -/
def wrapperExpectedMemory (P : Primitives α) (d : Storage) (p : Parameters)
    (a : Arguments) (initial : State α) : RegionName → Nat → Cell α :=
  if p.tokens = 1 then (zeros p initial).mem
  else expectedMemory P d (wrapperParameters p a) (kernelInitial p a initial)

theorem wrong_score_shape (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (a : Arguments) (initial : State α)
    (h : a.scoreRank ≠ 2) :
    invoke rest P d p a initial = none := by simp [invoke, h]

theorem single_token (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (a : Arguments) (initial : State α)
    (shape : a.scoreRank = 2) (token : p.tokens = 1) :
    invoke rest P d p a initial = some ⟨zeros p initial, [p.batches, 1]⟩ ∧
    (∀ o, o < p.batches → (zeros p initial).mem p.output o = .mk .int 0) ∧
    (∀ r o, (r ≠ p.output ∨ p.batches ≤ o) →
      (zeros p initial).mem r o = initial.mem r o) := by
  refine ⟨by simp [invoke, shape, token], ?_, ?_⟩
  · intro o ho; simp [zeros, ho]
  · intro r o h
    simp only [zeros]
    rw [if_neg (by rintro ⟨he, ho⟩; rcases h with h | h; exact h he; omega)]

theorem wrong_index_shape (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (a : Arguments) (initial : State α)
    (score : a.scoreRank = 2) (token : p.tokens ≠ 1)
    (bad : a.indexShape.any (fun sh => sh != [p.batches, p.tokens]) = Bool.true) :
    invoke rest P d p a initial = none := by simp [invoke, score, token, bad]

theorem empty_tokens (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (a : Arguments) (initial : State α) (token : p.tokens = 0) :
    invoke rest P d p a initial = none := by
  simp only [invoke, token, Nat.zero_ne_one, ↓reduceIte]
  split <;> simp

theorem default_indices (p : Parameters) (a : Arguments) (initial : State α)
    (omitted : a.indexShape = none) (o t : Nat) (ho : o < p.batches) (ht : t < p.tokens) :
    (prepareIndices p a initial).mem p.indices
      (o * (wrapperParameters p a).strideIndexBatch + t * (wrapperParameters p a).strideIndexToken) =
        .mk .int (Int.ofNat t) := by
  by_cases hb : p.batches = 1
  · have hz : o = 0 := by omega
    simp [prepareIndices, wrapperParameters, omitted, ht, hz]
  · simp [prepareIndices, wrapperParameters, omitted, ht, hb]

theorem clone_readback (p : Parameters) (a : Arguments) (initial : State α)
    (o : Nat) (ho : o < p.batches) :
    (kernelInitial p a initial).mem p.output o =
      .mk .int (((prepareIndices p a initial).mem p.indices
        (o * (wrapperParameters p a).strideIndexBatch)).read .int) := by
  simp [kernelInitial, cloneFirstColumn, wrapperParameters, ho]

theorem initialized_frame (p : Parameters) (a : Arguments) (initial : State α)
    (r : RegionName) (o : Nat)
    (hout : r ≠ p.output ∨ p.batches ≤ o)
    (hindex : a.indexShape.isNone = Bool.false ∨ r ≠ p.indices ∨ p.tokens ≤ o) :
    (kernelInitial p a initial).mem r o = initial.mem r o := by
  have hno : ¬ (r = p.output ∧ o < p.batches) := by
    rintro ⟨he, ho⟩
    rcases hout with h | h
    · exact h he
    · omega
  simp only [kernelInitial, cloneFirstColumn, wrapperParameters, hno, ↓reduceIte]
  unfold prepareIndices
  split
  · dsimp only
    split
    · rename_i hon hwrite
      rcases hindex with h | h | h
      · simp_all
      · exact (h hwrite.1).elim
      · omega
    · rfl
  · rfl

theorem wrapper_correct (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (a : Arguments) (initial : State α)
    (score : a.scoreRank = 2)
    (nonempty : 0 < p.tokens)
    (index : p.tokens ≠ 1 → a.indexShape.any (fun sh => sh != [p.batches, p.tokens]) = Bool.false)
    (fresh : FreshOutput p)
    (seeds : p.tokens ≠ 1 → ∀ k, k < grid (wrapperParameters p a) →
      SeedDomain (wrapperParameters p a)
        (programState (wrapperParameters p a) (kernelInitial p a initial) (kernelInitial p a initial) k)) :
    ∃ result, invoke rest P d p a initial = some result ∧
      result.state.mem = wrapperExpectedMemory P d p a initial ∧
      result.shape = if p.tokens = 1 then [p.batches, 1] else [p.batches] := by
  by_cases token : p.tokens = 1
  · refine ⟨⟨zeros p initial, [p.batches, 1]⟩, ?_, ?_, ?_⟩ <;>
      simp [invoke, score, token, wrapperExpectedMemory]
  · have hf : FreshOutput (wrapperParameters p a) := fresh
    obtain ⟨final, he, hm⟩ := launch_correct rest P d (wrapperParameters p a)
      (kernelInitial p a initial) hf (seeds token)
    refine ⟨⟨final, [p.batches]⟩, ?_, ?_, ?_⟩
    · simp [invoke, score, token, index token, Nat.ne_of_gt nonempty, he]
    · simpa only [wrapperExpectedMemory, token, ↓reduceIte] using hm
    · simp [token]

#specNonCircular wrapperExpectedMemory avoiding [invoke, run, runPrefix, FP.Structural.exec, source]
#axiomsClean single_token
#axiomsClean default_indices
#axiomsClean clone_readback
#axiomsClean initialized_frame
#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue.Host

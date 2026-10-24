import bench.repairs.tritonbench_g.multinomial_sampling.Contract
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Model

namespace VeriTile.Bench.Repairs.MultinomialSampling.Admission
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open Contract
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000

structure Observations (p : Parameters) where
  scoreFlags : Flags
  indexFlags : Flags
  seedFlags : Flags
  offsetFlags : Flags
  finite : Fin p.batches → Fin p.tokens → Bool
  units : Fin p.batches → Fin p.tokens → Int

def check {α Store Byte : Type} [Inhabited α] (api : Physical.API α Store Byte)
    (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (o : Observations p) : Bool :=
  ordinary o.scoreFlags && decide (call.scores.dtype = .fp32) &&
  decide (a.scoreRank = 2 ∧ call.scores.shape = [p.batches,p.tokens] ∧
    call.scores.strides = [p.strideScoreBatch,p.strideScoreToken]) &&
  shapeCheck p.batches p.tokens &&
  decide (a.indexShape.any (fun sh => sh != [p.batches,p.tokens]) = Bool.false) &&
  (if a.indexShape.isNone then Bool.true else
    ordinary o.indexFlags && decide (call.indices.dtype = .int64 ∧
      call.indices.device = call.scores.device ∧ call.indices.shape = [p.batches,p.tokens])) &&
  decide (∀ r, probabilityRow p.tokens (o.finite r) (o.units r) = Bool.true) &&
  (if p.tokens = 1 then Bool.true else
    ordinary o.seedFlags && ordinary o.offsetFlags &&
    decide (call.seeds.dtype = .int64 ∧ call.offsets.dtype = .int64 ∧
      call.seeds.device = call.scores.device ∧ call.offsets.device = call.scores.device ∧
      call.seeds.shape = [p.batches] ∧ call.offsets.shape = [p.batches] ∧
      a.seedsContiguous = Bool.true ∧ a.offsetsContiguous = Bool.true) &&
    decide (paddedRows p.batches < 2^31 ∧ paddedTokens p.tokens < 2^31) &&
    matrixWindow (paddedRows p.batches) (paddedTokens p.tokens)
      p.strideScoreBatch p.strideScoreToken call.scores.base 4 &&
    matrixWindow (paddedRows p.batches) (paddedTokens p.tokens)
      (Host.params p a).strideIndexBatch (Host.params p a).strideIndexToken call.indices.base 8 &&
    vectorWindow (paddedRows p.batches) call.seeds.base &&
    vectorWindow (paddedRows p.batches) call.offsets.base &&
    vectorWindow (paddedRows p.batches) call.output.base)

/-- Binding of the Python CPU readback/ratio observations to this call's
initial FP32 values. It concerns only individual input reads. -/
def Readback (p : Parameters) (s : State ℝ) (o : Observations p) : Prop :=
  ∀ r j, Host.inputWeights p s r.val j.val = (o.units r j : ℝ)/(2:ℝ)^149

theorem checked_valid {α Store Byte : Type} [Inhabited α]
    (api : Physical.API α Store Byte) (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (o : Observations p)
    (h : check api p a s call o = Bool.true) : Host.Valid p a := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at h
  have hb := h.1.1.1.1.2
  simp only [shapeCheck, decide_eq_true_eq] at hb
  refine ⟨h.1.1.1.1.1.2.1,h.1.1.1.2,hb.2.1,?_⟩
  intro ht
  simp only [ht, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.2.1.1.1.1.1.1.2.2.2.2.2.2.2.1,h.2.1.1.1.1.1.1.2.2.2.2.2.2.2.2⟩

theorem checked_rows {α Store Byte : Type} [Inhabited α]
    (api : Physical.API α Store Byte) (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (o : Observations p)
    (h : check api p a s call o = Bool.true) :
    ∀ r, probabilityRow p.tokens (o.finite r) (o.units r) = Bool.true := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1.2

theorem checked_domain {α Store Byte : Type} [Inhabited α]
    (api : Physical.API α Store Byte) (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (o : Observations p) (real : State ℝ)
    (h : check api p a s call o = Bool.true) (bind : Readback p real o) :
    ∀ r, r < p.batches → Spec.Domain p.tokens (Host.inputWeights p real r) := by
  intro r hr
  have hv := checked_valid api p a s call o h
  have hu := checked_rows api p a s call o h ⟨r,hr⟩
  simp only [probabilityRow, decide_eq_true_eq] at hu
  refine ⟨hv.2.2.1,?_,?_⟩
  · intro j hj
    rw [bind ⟨r,hr⟩ ⟨j,hj⟩]
    exact div_nonneg (Int.cast_nonneg (hu.1 ⟨j,hj⟩).2) (by positivity)
  · unfold Spec.cumulative
    rw [← Fin.sum_univ_eq_sum_range]
    simp_rw [bind ⟨r,hr⟩]
    rw [← Finset.sum_div,← Int.cast_sum,hu.2]
    norm_num

theorem window_index {rows tokens sb st base width r t : Nat}
    (h : matrixWindow rows tokens sb st base width = Bool.true)
    (hr : r < rows) (ht : t < tokens) :
    r*sb+t*st < 2^31 ∧ base+width*(r*sb+t*st) < 2^64 := by
  simp only [matrixWindow,decide_eq_true_eq] at h
  have hm : r*sb+t*st ≤ last rows tokens sb st := by
    unfold last
    exact Nat.add_le_add (Nat.mul_le_mul_right sb (by omega)) (Nat.mul_le_mul_right st (by omega))
  exact ⟨lt_of_le_of_lt hm h.2.2.1,
    lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left width hm) base) h.2.2.2.2⟩

theorem vector_index {rows base r : Nat}
    (h : vectorWindow rows base = Bool.true) (hr : r < rows) : base+8*r < 2^64 := by
  simp only [vectorWindow,decide_eq_true_eq] at h
  omega

theorem checked_abi {α Store Byte : Type} [Inhabited α]
    (api : Physical.API α Store Byte) (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (o : Observations p)
    (h : check api p a s call o = Bool.true) (ht : p.tokens ≠ 1) : IndexABI api p a s call := by
  have hshape : p.batches < 2^31 ∧ 0 < p.tokens ∧ p.tokens < 2^31 := by
    simp only [check,Bool.and_eq_true,decide_eq_true_eq] at h
    simpa only [shapeCheck,decide_eq_true_eq] using h.1.1.1.1.2
  have launch := h
  simp only [check,ht,↓reduceIte,Bool.and_eq_true,decide_eq_true_eq] at launch
  rcases launch.2 with ⟨⟨⟨⟨⟨⟨_,hp⟩,hs⟩,hi⟩,hseed⟩,hoff⟩,hout⟩
  have hss := hs
  have his := hi
  have hseed' := hseed
  have hoff' := hoff
  have hout' := hout
  simp only [matrixWindow,decide_eq_true_eq] at hss his
  simp only [vectorWindow,decide_eq_true_eq] at hseed' hoff' hout'
  refine ⟨hshape.1,hshape.2.2,hss.1,hss.2.1,his.1,his.2.1,
    hss.2.2.2.1,his.2.2.2.1,hseed'.1,hoff'.1,hout'.1,?_⟩
  intro r hr
  have hr' : r < paddedRows p.batches := hr
  refine ⟨lt_trans hr' hp.1,vector_index hseed hr',vector_index hoff hr',vector_index hout hr',?_⟩
  intro t htt
  have ht' : t < paddedTokens p.tokens := htt
  have sr := window_index hs hr' ht'
  have ir := window_index hi hr' ht'
  exact ⟨lt_trans ht' hp.2,sr.1,ir.1,sr.2,ir.2⟩

theorem signed64 (w : BitVec 64) : Random.IntegerType.i64.inRange w.toInt = Bool.true := by
  simp only [Random.IntegerType.inRange,decide_eq_true_eq]
  have lo := BitVec.le_toInt w
  have hi := BitVec.toInt_lt (x:=w)
  norm_num at lo hi ⊢
  omega

/-- Masked loads have arbitrary signed64 payloads, as required by the source
load type; they are never restricted to a chosen RNG seed. -/
def TypedUndefined {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  ∀ z, ∃ w : BitVec 64, s.undef .int p.seeds z = w.toInt

theorem checked_seeds {α Store Byte : Type} [Inhabited α]
    (api : Physical.API α Store Byte) (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (fresh : Host.Fresh p a)
    (undef : TypedUndefined p s) (pid : Nat) :
    SeedDomain (Host.params p a)
      (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState
        (Host.params p a) (Host.prepared p a s) (Host.prepared p a s) pid) := by
  intro i
  unfold seedValue intLoad
  split
  · rename_i ha
    have activeRow : row (Host.params p a)
        (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState
          (Host.params p a) (Host.prepared p a s) (Host.prepared p a s) pid) i < p.batches := by
      simpa only [decide_eq_true_eq,active] using ha
    change Random.IntegerType.i64.inRange
      (((Host.prepared p a s).mem p.seeds _).read .int) = Bool.true
    rw [Host.prepared_input p a s p.seeds fresh.1.2.1 (fun h => (fresh.2 h).2.1),call.seedInput _ activeRow]
    exact signed64 _
  · change Random.IntegerType.i64.inRange ((Host.prepared p a s).undef .int p.seeds _) = Bool.true
    have hu : (Host.prepared p a s).undef = s.undef := by
      unfold Host.prepared VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.kernelInitial
        VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.cloneFirstColumn
        VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.prepareIndices
      split <;> rfl
    rw [hu]
    obtain ⟨w,hw⟩ := undef _
    rw [hw]
    exact signed64 w

/-- Selected-label pointer totalization is independent of every FP primitive,
including scan rounding, comparisons and the argmax result. -/
theorem checked_label_read {α Store Byte : Type} [Inhabited α]
    (api : Physical.API α Store Byte) (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (o : Observations p)
    (h : check api p a s call o = Bool.true) (r position : Nat) (hr : r < p.batches) :
    min position (p.tokens-1) < p.tokens ∧
    call.indices.base+8*(r*(Host.params p a).strideIndexBatch+
      min position (p.tokens-1)*(Host.params p a).strideIndexToken)+8 ≤ call.indices.allocationEnd ∧
    call.indices.allocationEnd ≤ 2^64 := by
  have positive := (checked_valid api p a s call o h).2.2.1
  have hmin := Nat.min_le_right position (p.tokens-1)
  have bound : min position (p.tokens-1) < p.tokens := by omega
  exact ⟨bound,call.indexBounds r _ hr bound⟩

#axiomsClean checked_label_read
#axiomsClean checked_valid
#axiomsClean checked_domain
#axiomsClean checked_abi
#axiomsClean checked_seeds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Admission

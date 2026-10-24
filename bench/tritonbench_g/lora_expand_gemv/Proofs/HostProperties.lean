import bench.tritonbench_g.lora_expand_gemv.Proofs.HostMetadata
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 4000000

@[simp] theorem get_eq_ok (v : Option Nat) (e : HostError) (n : Nat) :
    get v e = .ok n ↔ v = some n := by
  cases v <;> simp [get]

structure Prepared (a : Arguments) (l : Launch) : Prop where
  inputDType : checkInput a.inputs.dtype = .ok l.inputType
  weightDType : checkWeight a.weights.dtype = .ok l.weightType
  matchingInnerSizes : ∃ k, a.inputs.size? 1 = some k ∧ a.weights.lastSize? = some k
  weightShape : ∃ sk sn tail, l.weightView.dimensions.reverse =
    (l.parameters.inner, sk) :: (l.parameters.outputSize, sn) :: tail
  inputContiguous : a.inputs.contiguous = Bool.true
  outputContiguous : a.output.contiguous = Bool.true
  weightContiguous : l.weightView.contiguous = Bool.true
  weightView :
    (a.weights.rank = 4 ∧ a.weights.size? 1 = some 1 ∧ l.weightView = a.weights.squeezeOne) ∨
    (a.weights.rank = 3 ∧ l.weightView = a.weights)
  indicesBatches : a.indices.size? 0 = some l.batches
  inputStrides : a.inputs.stride? 0 = some l.parameters.inputBatchStride ∧
    a.inputs.stride? 1 = some l.parameters.inputInnerStride
  weightStrides : l.weightView.stride? 0 = some l.parameters.loraStride ∧
    l.weightView.stride? 1 = some l.parameters.weightRowStride ∧
    l.weightView.stride? 2 = some l.parameters.weightInnerStride
  outputStrides : a.output.stride? 0 = some l.parameters.outputBatchStride ∧
    a.output.stride? 1 = some l.parameters.outputStride
  blockInnerPositive : 0 < l.parameters.blockInner
  blockRows : l.parameters.blockRows = 256
  splits : l.parameters.splits = l.parameters.outputSize
  blockInner : l.parameters.blockInner = nextPowerOfTwo l.parameters.inner
  evenK : l.parameters.evenK = decide (l.parameters.inner % l.parameters.blockInner = 0)
  castType : l.parameters.castType = chooseCast l.inputType
  addInputs : l.parameters.addInputs = a.addInputs
  outputType : l.outputType = a.output.dtype
  indexType : l.indexType = a.indices.dtype
  inputRegion : l.parameters.input = a.inputs.region
  outputRegion : l.parameters.output = a.output.region
  weightRegion : l.parameters.weight = a.weights.region
  indexRegion : l.parameters.indices = a.indices.region.cast

/-- Every successful host branch supplies the exact original launch choices;
none is an independently chosen kernel-specialization assumption. -/
theorem prepare_fields (a : Arguments) (l : Launch) (h : prepare a = .ok l) :
    Prepared a l := by
  unfold prepare at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' (first | split at h | simp [require] at h)
  all_goals cases h
  all_goals constructor <;> simp_all [require, Tensor.squeezeOne]
  all_goals omega

/-- The source's initial size equality and the post-squeeze K extraction
refer to the same dimension, including the 4D branch. -/
theorem prepared_inner_dimension (a : Arguments) (l : Launch)
    (h : prepare a = .ok l) :
    a.inputs.size? 1 = some l.parameters.inner ∧
      a.weights.lastSize? = some l.parameters.inner := by
  have hp := prepare_fields a l h
  obtain ⟨k, hi, hw⟩ := hp.matchingInnerSizes
  obtain ⟨sk, sn, tail, hshape⟩ := hp.weightShape
  have hl : l.weightView.lastSize? = some l.parameters.inner := by
    unfold Tensor.lastSize?
    rw [List.getLast?_eq_head?_reverse, hshape]
    rfl
  have hv : l.weightView.lastSize? = a.weights.lastSize? := by
    rcases hp.weightView with ⟨hr, _, hv⟩ | ⟨_, hv⟩
    · rw [hv]
      change (a.weights.dimensions.eraseIdx 1).getLast?.map Prod.fst =
        a.weights.dimensions.getLast?.map Prod.fst
      obtain ⟨d0, d1, d2, d3, hd⟩ := List.length_eq_four.mp hr
      simp [hd]
    · rw [hv]
  rw [hv, hw] at hl
  have hk := Option.some.inj hl
  exact ⟨hk ▸ hi, hk ▸ hw⟩

theorem prepare_positive_block (a : Arguments) (l : Launch) (h : prepare a = .ok l) :
    0 < l.parameters.blockRows := by
  rw [(prepare_fields a l h).blockRows]
  decide

theorem prepared_split_length_one (a : Arguments) (l : Launch)
    (h : prepare a = .ok l) (hn : 0 < l.parameters.outputSize) :
    Branches.length l.parameters = 1 := by
  have hs := (prepare_fields a l h).splits
  unfold Branches.length
  rw [hs]
  have he : l.parameters.outputSize + l.parameters.outputSize - 1 =
      (l.parameters.outputSize - 1) + 1 * l.parameters.outputSize := by omega
  rw [he, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt (by omega)]

#axiomsClean prepare_fields
#axiomsClean prepared_inner_dimension
#axiomsClean prepare_positive_block
#axiomsClean prepared_split_length_one
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host

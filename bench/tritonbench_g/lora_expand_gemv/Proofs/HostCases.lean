import bench.tritonbench_g.lora_expand_gemv.Proofs.HostProperties
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- Every metadata input either takes its ordered error path or yields a
launch certified from the actual wrapper computations. -/
theorem prepare_exhaustive (a : Arguments) :
    (∃ e, prepare a = .error e) ∨
      (∃ l, prepare a = .ok l ∧ Prepared a l) := by
  cases h : prepare a with
  | error e => exact Or.inl ⟨e,rfl⟩
  | ok l => exact Or.inr ⟨l,rfl,prepare_fields a l h⟩

theorem input_dtype_rejected (a : Arguments) (tag : String)
    (h : a.inputs.dtype = .other tag) : prepare a = .error .inputDTypeAssert := by
  simp [prepare, h, checkInput, bind, Except.bind, pure, Except.pure]

theorem weight_dtype_rejected (a : Arguments) (it : InputDType)
    (hi : checkInput a.inputs.dtype = .ok it)
    (hw : checkWeight a.weights.dtype = .error .weightDTypeAssert) :
    prepare a = .error .weightDTypeAssert := by
  simp [prepare, hi, hw, bind, Except.bind, pure, Except.pure]

/-- A 4D singleton axis is removed without changing any later host decision,
address stride, dtype choice, launch or rejection. Covers both ADD_INPUTS
values and every accepted input/weight format without specializing them. -/
theorem squeeze_branch_equivalent (a : Arguments) (r : RegionName) (d : DType)
    (l sl singletonStride n sn k sk : Nat) :
    prepare { a with weights := ⟨r,d,[(l,sl),(1,singletonStride),(n,sn),(k,sk)]⟩ } =
      prepare { a with weights := ⟨r,d,[(l,sl),(n,sn),(k,sk)]⟩ } := by
  simp [prepare, Tensor.rank, Tensor.size?, Tensor.lastSize?, Tensor.squeezeOne,
    Tensor.stride?, require, get, bind, Except.bind, pure, Except.pure]

theorem singleton_assert_rejected (a : Arguments) (it : InputDType) (wt : WeightDType)
    (k singleton : Nat)
    (hi : checkInput a.inputs.dtype = .ok it) (hw : checkWeight a.weights.dtype = .ok wt)
    (hix : a.inputs.size? 1 = some k) (hwk : a.weights.lastSize? = some k)
    (hic : a.inputs.contiguous = Bool.true) (hoc : a.output.contiguous = Bool.true)
    (hr : a.weights.rank = 4) (hs : a.weights.size? 1 = some singleton) (hn : singleton ≠ 1) :
    prepare a = .error .weightSingletonAssert := by
  simp [prepare, hi, hw, hix, hwk, hic, hoc, hr, hs, hn, require, get, bind, Except.bind, pure, Except.pure]

theorem rank_assert_rejected (a : Arguments) (it : InputDType) (wt : WeightDType) (k : Nat)
    (hi : checkInput a.inputs.dtype = .ok it) (hw : checkWeight a.weights.dtype = .ok wt)
    (hix : a.inputs.size? 1 = some k) (hwk : a.weights.lastSize? = some k)
    (hic : a.inputs.contiguous = Bool.true) (hoc : a.output.contiguous = Bool.true)
    (hfour : a.weights.rank ≠ 4) (hthree : a.weights.rank ≠ 3) :
    prepare a = .error .weightRankAssert := by
  simp [prepare, hi, hw, hix, hwk, hic, hoc, hfour, hthree, require, get, bind, Except.bind, pure, Except.pure]

/-- A genuine original-source failure: accepted zero-K metadata reaches
next_power_of_2(0)=0, then Python's K % BLOCK_K raises before index-size or
stride argument access and before any kernel launch. The 4D case follows
through squeeze_branch_equivalent. -/
theorem zero_inner_rejected (a : Arguments) (it : InputDType) (wt : WeightDType)
    (n sk sn : Nat) (tail : List (Nat × Nat))
    (hi : checkInput a.inputs.dtype = .ok it) (hw : checkWeight a.weights.dtype = .ok wt)
    (hix : a.inputs.size? 1 = some 0) (hwk : a.weights.lastSize? = some 0)
    (hic : a.inputs.contiguous = Bool.true) (hoc : a.output.contiguous = Bool.true)
    (hr : a.weights.rank = 3) (hwc : a.weights.contiguous = Bool.true)
    (hshape : a.weights.dimensions.reverse = (0,sk) :: (n,sn) :: tail) :
    prepare a = .error .evenKZeroDivision := by
  simp [prepare, hi, hw, hix, hwk, hic, hoc, hr, hwc, hshape,
    nextPowerOfTwo, require, get, bind, Except.bind, pure, Except.pure]

#axiomsClean prepare_exhaustive
#axiomsClean input_dtype_rejected
#axiomsClean weight_dtype_rejected
#axiomsClean squeeze_branch_equivalent
#axiomsClean singleton_assert_rejected
#axiomsClean rank_assert_rejected
#axiomsClean zero_inner_rejected
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host

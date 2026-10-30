import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.APIWords

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

abbrev FP32Word := VeriTile.Triton.FP.Value .fp32

/-! The native profile is FP32 throughout, allow_tf32=False, with finite SIMT
multiply/add/FMA dot syntax and live accumulator; host sum(0) is a separate
finite addition tree followed by the actual cast. The saved SM90 artifacts
contain no tensor-core MMA. The scalar ports preserve scale placement, casts,
causal masking, previous/future/diagonal order and the scalar log/exp calls.

ScalarAgreement contains only constants, conversion, scalar arithmetic,
FMA, log2, exp2, cast and the host D**-0.5 conversion. It contains no dot,
reduction, complete-output, successful-run or memory-frame premise. No such
numerical law is discharged here. Native exp2 includes ex2.approx.ftz.f32;
the log2 lowering contains its own FTZ polynomial arithmetic. Its numerical
contract belongs to that elementary scalar port. Treating these operations
as exact real exp/log, or substituting an unrecorded dot/reduction tree, is
not justified by this theorem. The retained compiler/AST evidence pins the
source lowering separately from local numerical conformance.
-/

/-- Complete returned FP32 word tensors under scalar-level correspondence.
The reference evaluates the retained finite expression trees, not the real
causal formula. Both forwards and the shared backward state read the bound
logical public inputs and actual scratch contents. -/
specification fp32_public_words (abi : LocalRuntime FP32Word σ)
    (native reference : Ops FP32Word) (primitives : ScalarAgreement native reference)
    (p : Plans) (s : Shape) (_domain : ShapeContract s) (b : Bases) (tree : SumPlan) (trees : OutputKind → SumPlan)
    (forwardLayout : ForwardLayout s b) (backwardLayout : BackwardLayout s b)
    (saved : SavedInputs s b) (forwardState backwardState : σ) (x : LogicalInputs FP32Word s)
    (forwardInputs : ForwardMaterialized s b (abi.observe forwardState) x)
    (backwardInputs : Materialized s b (abi.observe backwardState) x) :
    let f := forwardAPI abi native p s b tree forwardLayout saved forwardState
    let g := backwardAPI abi native p s b trees backwardLayout f.context backwardState
    (∀ i,readReturned s .forward f.output (abi.observe f.state) i=
      wordReference reference p s .forward tree (x.padded reference)
        (dense s.T s.DV i.bh i.time i.feature)) ∧
    (∀ kind,kind≠.forward → ∀ i,readReturned s kind (g.handle kind) (abi.observe g.state) i=
      wordReference reference p s kind (trees kind) (x.padded reference)
        (dense s.T (kind.width s) i.bh i.time i.feature)) ∧
    (∀ address,
      (address<forwardLayout.scratch ∨ forwardLayout.scratch+OutputKind.forward.count s≤address) →
      (address<forwardLayout.output ∨ forwardLayout.output+OutputKind.forward.cells s≤address) →
      abi.observe f.state address=abi.observe forwardState address) ∧
    (∀ address,
      (∀ cell∈allCells s true,cellsAddress backwardLayout.scratch cell≠address) →
      (∀ cell∈allCells s false,cellsAddress backwardLayout.output cell≠address) →
      abi.observe g.state address=abi.observe backwardState address) := by
  rw [primitives.eq]
  dsimp only
  refine ⟨?_,?_,?_,?_⟩
  · intro i
    rw [forward_api_word]
    have inputs := materialized_forward_reads reference s b (abi.observe forwardState) x forwardInputs
    exact forward_word_inputs reference p s tree _ _ inputs.1 inputs.2.1 inputs.2.2 _
  · intro kind active i
    rw [backward_api_word abi reference p s b trees backwardLayout _ backwardState kind active,
      materialized_read reference s b (abi.observe backwardState) x backwardInputs]
  · exact forward_api_frame abi reference p s b tree forwardLayout saved forwardState
  · exact backward_api_frame abi reference p s b trees backwardLayout _ backwardState

/-- Bitwise primitive equality lifts through the complete API state, including
all output buffers and the unchanged memory complement. No real-number
reassociation, dot oracle or assumed complete launch result is involved. -/
specification fp32_backend_equiv (abi : LocalRuntime FP32Word σ)
    (native reference : Ops FP32Word) (primitives : ScalarAgreement native reference)
    (p : Plans) (s : Shape) (_domain : ShapeContract s) (b : Bases) (tree : SumPlan) (trees : OutputKind → SumPlan)
    (forwardLayout : ForwardLayout s b) (backwardLayout : BackwardLayout s b)
    (saved : SavedInputs s b) (context : SavedContext) (state : σ) :
    forwardAPI abi native p s b tree forwardLayout saved state=
      forwardAPI abi reference p s b tree forwardLayout saved state ∧
    backwardAPI abi native p s b trees backwardLayout context state=
      backwardAPI abi reference p s b trees backwardLayout context state := by
  rw [primitives.eq]
  exact ⟨rfl,rfl⟩

#axiomsClean fp32_public_words
#axiomsClean fp32_backend_equiv
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention

import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalInputs

/-! Exact checked FP32 domain, complete mathematical forward/backward IO,
source host wiring, saved producer statistics and native index safety. Ordinary
successful contiguous/view/allocator/compiler/runtime APIs remain explicit. -/
namespace VeriTile.Bench.Repairs.LayerNormLiger.Correctness
open VeriTile.Triton FP.Structural VeriTile.Torch
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
open Logical
noncomputable section

abbrev python_checks_iff_preconditions := Contract.checkArguments_iff
abbrev python_gradient_check_iff := Contract.checkGradient_iff
abbrev preparation_preserves_original_values := @Preparation.complete
abbrev prepared_original_input_cells := @Preparation.input_cells
abbrev native_row_arithmetic := NativeIndex.source_row
abbrev native_group_arithmetic := NativeIndex.source_group

/-- All wrapper-generated parameters, allocation order, source group count,
forward-produced saved statistics and returned gradient reductions are attached
to the old complete theorem. No arbitrary kernel parameters remain public. -/
specification layer_norm_roundtrip_real (rest : Algebra ℝ) (s : State ℝ)
    (x w b dy : TensorMetadata) (eps : Contract.Epsilon) (r : Wrapper.Inputs) (a : Wrapper.Allocations)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (h : Contract.InputContract x w b eps) (g : Contract.GradientContract dy x)
    (hp : 0 < programs) (fresh : Wrapper.Fresh r a)
    (storage : Wrapper.Storage (Contract.rows x) (Contract.width x) programs) :
    NativeIndex.Safe (Contract.rows x) (Contract.width x) programs (Settings.block (Contract.width x)) ∧
    ∃ f out,
      Grid.run (withSum (algebra rest) sumPrimitive) (Forward.kernel (Wrapper.forward r a x eps)) s
        (Contract.rows x) = some f ∧
      Wrapper.checked (algebra rest) sumPrimitive Returned.scalar x w b dy eps r a programs pw pb s = some out ∧
      Public.Correct (Wrapper.backward r a x programs) f programs (a.region 6) (a.region 7) dy.shape out ∧
      (∀ row, row < Contract.rows x →
        Backward.mean (Wrapper.backward r a x programs) (saved (Wrapper.backward r a x programs) f) row =
          mean (Contract.width x) (rowX (Wrapper.forward r a x eps) (atPid s row)) ∧
        Backward.rstd (Wrapper.backward r a x programs) (saved (Wrapper.backward r a x programs) f) row =
          rstd (Contract.width x) eps.value (rowX (Wrapper.forward r a x eps) (atPid s row))) ∧
      (∀ row, row < Contract.rows x → ∀ i : TileIndex [Settings.block (Contract.width x)],
        i.1.val < Contract.width x →
        f.mem (a.region 0) (row * Contract.width x + i.1.val) = .mk .fp32
          (y (Contract.width x) eps.value (rowX (Wrapper.forward r a x eps) (atPid s row))
            (weight (Wrapper.forward r a x eps) s) (bias (Wrapper.forward r a x eps) s) i.1.val)) := by
  refine ⟨Wrapper.native x w b eps programs h hp storage, ?_⟩
  rw [Wrapper.checked_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ h g hp]
  exact PublicReader.returned_logical rest (Wrapper.forward r a x eps) (Wrapper.backward r a x programs)
    s (Contract.rows x) programs pw pb (a.region 6) (a.region 7) dy.shape
    (Wrapper.layout r a x w b dy eps programs hp h g fresh) (Wrapper.link r a x eps programs fresh)

/-- Actual backward-call state. Its DY may alias Y or another saved read-only
input; fresh backward allocations separate only writes. This theorem applies
after ordinary autograd delivery/preparation of the incoming cotangent. -/
theorem layer_norm_backward_real (rest : Algebra ℝ) (s : State ℝ)
    (x w b dy : TensorMetadata) (eps : Contract.Epsilon) (r : Wrapper.Inputs) (a : Wrapper.Allocations)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (h : Contract.InputContract x w b eps) (g : Contract.GradientContract dy x)
    (hp : 0 < programs) (fresh : Wrapper.Fresh r a) :
    ∃ out, Host.backward (algebra rest) sumPrimitive Returned.scalar (Wrapper.backward r a x programs)
      programs pw pb (a.region 6) (a.region 7) dy.shape s = some out ∧
      Public.Correct (Wrapper.backward r a x programs) s programs (a.region 6) (a.region 7) dy.shape out :=
  Public.backward rest _ s _ _ _ _ _ _ (Wrapper.layout r a x w b dy eps programs hp h g fresh)

/-- Original-Tensor mathematical output after the exact successful API
preparation. Storage capacity is derived from that preparation and the actual
partial allocation, not introduced as another admission restriction. -/
specification layer_norm_original_inputs (rest : Algebra ℝ) (s : State ℝ)
    (lib : Preparation.Library ℝ) (laws : Preparation.Laws lib)
    (x w b dy : Preparation.Tensor ℝ) (eps : Contract.Epsilon)
    (p : Preparation.Prepared ℝ) (prepared : Preparation.run lib x w b dy = some p)
    (ready : Preparation.Ready s p) (a : Wrapper.Allocations)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (h : Contract.InputContract x.metadata w.metadata b.metadata eps)
    (g : Contract.GradientContract dy.metadata x.metadata) (hp : 0 < programs)
    (fresh : Wrapper.Fresh (Preparation.regions p) a)
    (partialBase : Nat) (partialSpan : partialBase + 4 * (programs * Contract.width x.metadata) ≤ 2^64) :
    NativeIndex.Safe (Contract.rows x.metadata) (Contract.width x.metadata) programs
      (Settings.block (Contract.width x.metadata)) ∧
    ∃ f out,
      Grid.run (withSum (algebra rest) sumPrimitive)
        (Forward.kernel (Wrapper.forward (Preparation.regions p) a x.metadata eps)) s (Contract.rows x.metadata) = some f ∧
      Wrapper.checked (algebra rest) sumPrimitive Returned.scalar x.metadata w.metadata b.metadata dy.metadata
        eps (Preparation.regions p) a programs pw pb s = some out ∧
      Public.Correct (Wrapper.backward (Preparation.regions p) a x.metadata programs) f programs
        (a.region 6) (a.region 7) dy.metadata.shape out ∧
      (∀ row, row < Contract.rows x.metadata → ∀ col, col < Contract.width x.metadata →
        f.mem (a.region 0) (row * Contract.width x.metadata + col) = .mk .fp32
          (y (Contract.width x.metadata) eps.value
            (fun j => x.logical (row * Contract.width x.metadata + j)) w.logical b.logical col)) := by
  have api := Preparation.complete lib laws x w b dy eps h g p prepared
  have inputSpan : 0 + 4 * (Contract.rows x.metadata * Contract.width x.metadata) ≤ 2^64 := by
    simpa only [Nat.zero_add,Contract.shape_elements x.metadata h.rank] using api.capacity
  obtain ⟨native,f,out,frun,run,correct,stats,output⟩ :=
    layer_norm_roundtrip_real rest s x.metadata w.metadata b.metadata dy.metadata eps (Preparation.regions p)
      a programs pw pb h g hp fresh ⟨0,partialBase,inputSpan,partialSpan⟩
  refine ⟨native,f,out,frun,run,correct,?_⟩
  intro row hr col hc
  have cb : col < Settings.block (Contract.width x.metadata) :=
    lt_of_lt_of_le hc (Settings.nextPower_covers _)
  have got := output row hr (⟨col,cb⟩,PUnit.unit) hc
  rw [LogicalInputs.y_congr (Contract.width x.metadata) col eps.value
    (rowX (Wrapper.forward (Preparation.regions p) a x.metadata eps) (atPid s row))
    (weight (Wrapper.forward (Preparation.regions p) a x.metadata eps) s)
    (bias (Wrapper.forward (Preparation.regions p) a x.metadata eps) s)
    (fun j => x.logical (row * Contract.width x.metadata + j)) w.logical b.logical hc
    (fun j hj => LogicalInputs.row s x w b dy p a eps h api ready row j hr hj)
    (fun j hj => LogicalInputs.weight s x w b dy p a eps h api ready j hj)
    (fun j hj => LogicalInputs.bias s x w b dy p a eps h api ready j hj)] at got
  exact got

#axiomsClean layer_norm_original_inputs
#axiomsClean layer_norm_roundtrip_real
#axiomsClean layer_norm_backward_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Correctness

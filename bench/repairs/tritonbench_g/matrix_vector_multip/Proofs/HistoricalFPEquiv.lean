import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.FP
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.TensorBinding
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Numerical
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.ReturnedTensor

/-! Default FP32 inputs, explicit FP32 casts and accumulator, FP32 output.
Every bit pattern is admitted. No associativity, multiply/add contraction
identity, numerical output premise, or FP-to-Real equality is assumed. -/
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.HistoricalFPEquiv
open VeriTile.Triton VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeHost WholeNative
noncomputable section

def precision : FloatDType := .fp32
abbrev bits (w : Torch.FP32.Word) : BitVec 32 := w.bits

/-- Independent ordered graph on original logical FP32 input words; every
masked zero, source-loop carry, FMA orientation and reduction edge is retained. -/
def orderedFormula := FP.formula

specification mv_fp32_outcomes
    (native : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (x y : Torch.FP32.Tensor) (lx : Torch.FP32.TensorLaws x) (ly : Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata) (read : Nat → Torch.FP32.Word) (old : Nat → Prop)
    (rt : Runtime (BitVec 32) (TensorBinding.descriptor x y lx ly h)
      (TensorBinding.memory bits read) old)
    (outputSpan : ∀ a, rt.allocate = some a → 4*a.base + 4*rows x.metadata ≤ 2^64)
    (launchable : ∀ a, rt.allocate = some a → rt.launchReady a = Bool.true →
      ∀ t, t ∈ (rt.plan a).trials ∨ t = (rt.plan a).selected →
        WholeGrid.gridSize ((TensorBinding.descriptor x y lx ly h).parameters t.config) < 2^31) :
    executeChecked (WholeNativeHost.execute native graphs) rt =
      .ok (WholeNativeHost.execute native graphs rt) ∧
    WholePublic.Outcomes (fun a i => orderedFormula native graphs (rt.plan a).selected.config
      (TensorBinding.descriptor x y lx ly h)
      (TensorBinding.matrix x lx bits read) (TensorBinding.vector y ly bits read) i.val)
      rt (WholeNativeHost.execute native graphs rt) ∧
    ∀ a, rt.allocate = some a → rt.launchReady a = Bool.true →
      Index.Correspondence a ∧
      Index.PlanCorrespondence (TensorBinding.descriptor x y lx ly h) (rt.plan a) := by
  refine ⟨checked_run h _ rt, ?_, ?_⟩
  · exact FP.outcomes native graphs (TensorBinding.direct x y lx ly h bits read) h.dimensions rt
  · intro a allocated ready
    have bounds := TensorBinding.bounds x y lx ly h a (outputSpan a allocated)
    exact ⟨Index.correspondence a bounds,
      Index.plan_correspondence ⟨bounds, launchable a allocated ready⟩⟩

/-- Successful source execution returns the exact fresh allocation Tensor.
The input-domain premise is exactly the Python checks; remaining premises
are ordinary Tensor/allocation/legal-launch bindings. -/
specification mv_fp32_returned_tensor
    (native : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (x y : Torch.FP32.Tensor) (lx : Torch.FP32.TensorLaws x) (ly : Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata) (read : Nat → Torch.FP32.Word) (old : Nat → Prop)
    (rt : Runtime (BitVec 32) (TensorBinding.descriptor x y lx ly h)
      (TensorBinding.memory bits read) old)
    (a : Allocated (BitVec 32) (TensorBinding.descriptor x y lx ly h)
      (TensorBinding.memory bits read) old)
    (allocated : rt.allocate = some a) (ready : rt.launchReady a = Bool.true)
    (allocation : ReturnedTensor.Allocation a)
    (launchable : ∀ t, t ∈ (rt.plan a).trials ∨ t = (rt.plan a).selected →
      WholeGrid.gridSize ((TensorBinding.descriptor x y lx ly h).parameters t.config) < 2^31) :
    ∃ finalMemory,
      (executeChecked (WholeNativeHost.execute native graphs) rt).map
        (ReturnedTensor.returnAllocated allocation.tensor) =
        .ok (some (allocation.tensor, finalMemory)) ∧
      (∀ i : Fin (rows x.metadata),
        (Torch.FP32.logicalValue (ReturnedTensor.readWords finalMemory)
          allocation.tensor allocation.laws i.val).bits =
        orderedFormula native graphs (rt.plan a).selected.config
          (TensorBinding.descriptor x y lx ly h)
          (TensorBinding.matrix x lx bits read) (TensorBinding.vector y ly bits read) i.val) ∧
      (∀ addr, old addr → finalMemory addr = TensorBinding.memory bits read addr) ∧
      (∀ addr, (∀ i, i < rows x.metadata → addr ≠ a.base+i) → finalMemory addr = a.memory addr) ∧
      Index.Correspondence a ∧
      Index.PlanCorrespondence (TensorBinding.descriptor x y lx ly h) (rt.plan a) ∧
      (0 < rows x.metadata → BitVec.ofNat 64 (4*a.base) = allocation.tensor.dataPtr) ∧
      allocation.tensor.metadata.shape = [rows x.metadata] ∧
      allocation.tensor.metadata.dtype = .floating .fp32 ∧
      allocation.tensor.metadata.device.kind = "cuda" ∧
      allocation.tensor.metadata.device.index = x.metadata.device.index := by
  obtain ⟨memory, returned, values, frame, exterior⟩ := ReturnedTensor.returns native graphs rt
    h.dimensions allocated ready allocation (TensorBinding.direct x y lx ly h bits read)
  have bounds := TensorBinding.bounds x y lx ly h a (ReturnedTensor.span allocation)
  refine ⟨memory, ?_, values, frame, exterior, Index.correspondence a bounds,
    Index.plan_correspondence ⟨bounds, launchable⟩, ReturnedTensor.pointer allocation,
    allocation.shape, allocation.laws.dtype, allocation.cuda, allocation.device⟩
  rw [checked_run h]
  simpa only [Except.map] using congrArg Except.ok returned

/-- Conditional numerical transport replaces each finite-operand FMA and add
with concrete IEEE binary32 RNE/non-FTZ semantics. NaN/Inf operands retain the
native functions. Admission is unchanged and no intermediate is assumed finite. -/
specification mv_fp32_finite_ops_if_local
    (native : Scalars) (numerical : Numerical.LocalFP32 native)
    (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (x y : Torch.FP32.Tensor) (lx : Torch.FP32.TensorLaws x) (ly : Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata) (read : Nat → Torch.FP32.Word) (old : Nat → Prop)
    (rt : Runtime (BitVec 32) (TensorBinding.descriptor x y lx ly h)
      (TensorBinding.memory bits read) old) :
    WholePublic.Outcomes (fun a i => orderedFormula (Numerical.finiteReference native)
      graphs (rt.plan a).selected.config (TensorBinding.descriptor x y lx ly h)
      (TensorBinding.matrix x lx bits read) (TensorBinding.vector y ly bits read) i.val)
      rt (WholeNativeHost.execute native graphs rt) := by
  have reference := FP.outcomes native graphs
    (TensorBinding.direct x y lx ly h bits read) h.dimensions rt
  have eqn := Numerical.native_eq_reference native numerical
  simpa only [orderedFormula, ←eqn] using reference

#axiomsClean mv_fp32_returned_tensor
#axiomsClean mv_fp32_outcomes
#axiomsClean mv_fp32_finite_ops_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.HistoricalFPEquiv

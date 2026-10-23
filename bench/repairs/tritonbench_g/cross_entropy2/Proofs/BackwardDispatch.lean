import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfileDispatch

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

structure BackwardKey where
  common : Scheduled.JITKey
  gradientDiv16 : Bool
  gradientStrideDiv16 : Bool
  upstreamStrideDiv16 : Bool
  deriving DecidableEq

def backwardKey {σ : Type} (layout : Layout σ) (state : World Compiled σ) (p : Parameters) : BackwardKey :=
  ⟨Scheduled.keyFor layout state p,
    decide (Scheduled.bytePointer layout state p.gradient p.gradientBase%16=0),
    decide (p.gradientStride%16=0),decide (p.upstreamStride%16=0)⟩

/-- The only per-element native backward choices left after the fixed scalar
lowering are factor/output commutations. The already-proved local scalar laws
interpret both choices, including all smoothing/hit/ignored branches. -/
structure BackwardOrder where
  fma : Nat → Bool
  output : Nat → Bool

abbrev BackwardABI := BackwardKey → BackwardOrder

structure BackwardLibrary {σ : Type} (layout : Layout σ) (abi : BackwardABI)
    (order : BackwardOrder) (rt : Runtime Compiled σ) : Prop where
  compile : ∀ p count before after,
    rt.setupBackward2 p count before=.ok () after → order=abi (backwardKey layout after p)

def BackwardSelected {σ : Type} (layout : Layout σ) (abi : BackwardABI) (order : BackwardOrder)
    {device : DeviceFamily} {rt : Runtime Compiled σ} {o : Options} {g x l y : Tensor}
    {rows cols : Nat} {initial final : World Compiled σ} {output : Tensor}
    (e : BackwardData (algebra device o.logitScale) rt o g x l y rows cols initial output final) : Prop :=
  rows=0 ∨ ∃ ready,
    rt.setupBackward2 e.parameters (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width) e.allocated=.ok () ready ∧
    final={ready with memory:=(backwardGrid (algebra device o.logitScale) e.parameters ready.memory
      (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width))} ∧
    order=abi (backwardKey layout ready e.parameters)

theorem source_backward_selected {σ : Type} {layout : Layout σ} {abi : BackwardABI} {order : BackwardOrder}
    {device : DeviceFamily} {rt : Runtime Compiled σ} {o : Options} {g x l y : Tensor}
    {rows cols : Nat} {initial final : World Compiled σ} {output : Tensor}
    (dispatch : BackwardLibrary layout abi order rt)
    (e : BackwardData (algebra device o.logitScale) rt o g x l y rows cols initial output final) :
    BackwardSelected layout abi order e := by
  let count := VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width
  have extract : ∀ before done,
      backwardLaunch (algebra device o.logitScale) rt e.parameters count before=.ok () done →
      rows=0 ∨ ∃ ready,
        rt.setupBackward2 e.parameters count before=.ok () ready ∧
        done={ready with memory:=backwardGrid (algebra device o.logitScale) e.parameters ready.memory count} ∧
        order=abi (backwardKey layout ready e.parameters) := by
    intro before done call
    obtain ⟨trace⟩ := launch_trace e.parameters.rows (rt.setupBackward2 e.parameters count)
      (fun memory => backwardGrid (algebra device o.logitScale) e.parameters memory count) before done call
    cases trace with
    | empty zero => exact Or.inl zero
    | dispatched nonzero ready setup => exact Or.inr ⟨ready,setup,rfl,dispatch.compile _ _ _ _ setup⟩
  exact extract _ _ e.launchCall

#axiomsClean source_backward_selected
#specNonCircular BackwardLibrary avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled

import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfiledReadback

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled
open VeriTile.Triton Source Host Reduction
open Scheduled (decode keyFor JITKey)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations IntegerViews LocalTrace)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Actual local TensorIterator inputs. Singleton/empty shapes, offsets and
strides remain visible; this is not a fixture-only shape label. -/
structure TorchMetadata where
  shape : List Nat
  strides : List Nat
  offset : Nat
  bytePointer : Nat
  device : Nat
  dtype : String
  deriving DecidableEq

def torchMetadata {σ : Type} (layout : Layout σ) (state : World Compiled σ) (input : Tensor) : TorchMetadata :=
  ⟨input.shape,input.strides,input.storageOffset,
    Scheduled.bytePointer layout state input.region input.storageOffset,input.device,input.dtype⟩

/-- One installed compiler/Torch configuration exports total graph-valued
dispatch APIs. Target, toolchain and launch defaults belong to this ordinary
backend configuration. No API field accepts tensor values or returns a Word.
Vector heads/tails, CTA stages and iterator splits are represented by the
returned explicit trees, with their full leaf permutation and seeds. -/
structure BackendABI where
  device : JITKey → DeviceFamily
  torchSum : TorchMetadata → ∀ width, 0<width → Plan width
  torchLSEMax : TorchMetadata → ∀ width, 0<width → Plan width
  torchLSESum : TorchMetadata → ∀ width, 0<width → Plan width

/-- Ordinary local dispatch contracts, separate from numerical obligations.
They relate actual successful compile/library calls to explicit plan data.
No clause mentions the cross-entropy output, whole host success, reduction
value, floating operands, or numerical equivalence of complete expressions.
The Torch clauses are for the source's dense FP32 [count,rows] allocations;
they expose the uniform per-output arithmetic graph of these local calls. -/
structure DispatchLibrary {σ : Type} (layout : Layout σ) (abi : BackendABI)
    (device : DeviceFamily) (host : Profile) (rt : Runtime Compiled σ) : Prop where
  compile : ∀ p count before after (positive : 0<p.block),
    rt.setupForward2 p count before=.ok () after →
      device p.block positive=abi.device (keyFor layout after p) p.block positive
  sum : ∀ input before output after width rows (positive : 0<width),
    input.shape=[width,rows] → input.Dense → input.dtype="torch.float32" →
    rt.sum0 input before=.ok output after →
      host.hostSumTree width positive=abi.torchSum (torchMetadata layout before input) width positive
  lse : ∀ input before output after width rows (positive : 0<width),
    input.shape=[width,rows] → input.Dense → input.dtype="torch.float32" →
    rt.logsumexp0 input before=.ok output after →
      host.hostMaxTree width positive=abi.torchLSEMax (torchMetadata layout before input) width positive ∧
      host.hostLSESumTree width positive=abi.torchLSESum (torchMetadata layout before input) width positive

variable {σ : Type} {layout : Layout σ} {abi : BackendABI} {device : DeviceFamily} {host : Profile}
  {obs : Observations Compiled σ} {rt : Runtime Compiled σ} {o : Options} {x y : Tensor}
  {rows cols : Nat} {initial final : World Compiled σ} {returned : Return}
  {e : ForwardData (algebra device o.logitScale) rt o x y rows cols initial returned final}

/-- Exact local source dispatch evidence. Empty grids and direct/non-split
paths do not fabricate compiler or Torch calls. -/
def DeviceSelected (layout : Layout σ) (abi : BackendABI) (device : DeviceFamily)
    (e : ForwardData (algebra device o.logitScale) rt o x y rows cols initial returned final) : Prop :=
  rows=0 ∨ ∃ ready,
    rt.setupForward2 e.parameters e.count e.zAllocated=.ok () ready ∧
    e.computed={ready with memory:=Grid.forward (algebra device o.logitScale) e.parameters ready.memory e.count} ∧
    ∀ positive : 0<e.parameters.block,
      device e.parameters.block positive=abi.device (keyFor layout ready e.parameters) e.parameters.block positive

def TorchSelected (layout : Layout σ) (abi : BackendABI) (host : Profile)
    (e : ForwardData (algebra device o.logitScale) rt o x y rows cols initial returned final) : Prop :=
  e.split=Bool.false ∨ e.count≤1 ∨ ∃ (localLSE localLoss : Tensor) (logged summed : World Compiled σ),
    rt.logsumexp0 e.lse e.computed=.ok localLSE logged ∧
    rt.sum0 e.loss logged=.ok localLoss summed ∧
    ∀ positive : 0<e.count,
      host.hostMaxTree e.count positive=abi.torchLSEMax (torchMetadata layout e.computed e.lse) e.count positive ∧
      host.hostLSESumTree e.count positive=abi.torchLSESum (torchMetadata layout e.computed e.lse) e.count positive ∧
      host.hostSumTree e.count positive=abi.torchSum (torchMetadata layout logged e.loss) e.count positive

theorem source_device_selected (dispatch : DispatchLibrary layout abi device host rt)
    (e : ForwardData (algebra device o.logitScale) rt o x y rows cols initial returned final) :
    DeviceSelected layout abi device e := by
  have extract : ∀ before computed,
      forwardLaunch (algebra device o.logitScale) rt e.parameters e.count before=.ok () computed →
      rows=0 ∨ ∃ ready,
        rt.setupForward2 e.parameters e.count before=.ok () ready ∧
        computed={ready with memory:=Grid.forward (algebra device o.logitScale) e.parameters ready.memory e.count} ∧
        ∀ positive : 0<e.parameters.block,
          device e.parameters.block positive=abi.device (keyFor layout ready e.parameters) e.parameters.block positive := by
    intro before computed call
    obtain ⟨trace⟩ := launch_trace e.parameters.rows (rt.setupForward2 e.parameters e.count)
      (fun memory => Grid.forward (algebra device o.logitScale) e.parameters memory e.count) before computed call
    cases trace with
    | empty zero => exact Or.inl zero
    | dispatched nonzero ready setup =>
      exact Or.inr ⟨ready,setup,rfl,fun positive => dispatch.compile _ _ _ _ positive setup⟩
  exact extract _ _ e.launchCall

theorem source_torch_selected (dispatch : DispatchLibrary layout abi device host rt)
    (k : ForwardKernelData obs e) : TorchSelected layout abi host e := by
  cases split : e.split with
  | «false» => exact Or.inl split
  | «true» =>
    have finish : aggregate rt o e.count e.labels e.loss e.lse e.computed=.ok returned final := by
      simpa only [split,↓reduceIte] using e.finishCall
    obtain ⟨a⟩ := aggregate_success rt o e.count e.labels e.loss e.lse e.computed returned final finish
    have extract : ∀ pair reduced,
        LocalTrace rt.toRuntime e.count e.loss e.lse e.computed pair reduced → TorchSelected layout abi host e := by
      intro pair reduced trace
      cases trace with
      | direct one => exact Or.inr (Or.inl (by omega))
      | reduced many localLSE localLoss logged summed lseCall sumCall =>
        refine Or.inr (Or.inr ⟨localLSE,localLoss,logged,_,lseCall,sumCall,?_⟩)
        intro positive
        have lseShape : e.lse.shape=[e.count,rows] := by
          simpa only [if_pos many] using k.lseAllocation.shapeEq
        have lossShape : e.loss.shape=[e.count,rows] := by
          simpa only [if_pos many] using k.lossAllocation.shapeEq
        have lse := dispatch.lse _ _ _ _ e.count rows positive lseShape k.lseDense k.lseAllocation.dtypeEq lseCall
        exact ⟨lse.1,lse.2,dispatch.sum _ _ _ _ e.count rows positive lossShape k.lossDense k.lossAllocation.dtypeEq sumCall⟩
    exact extract _ _ a.localCalls

/-- Every positive source width is handled by the total profile family.
Every dispatch branch comes from the actual source trace. -/
theorem source_profiles_selected (dispatch : DispatchLibrary layout abi device host rt)
    (k : ForwardKernelData obs e) :
    DeviceSelected layout abi device e ∧ TorchSelected layout abi host e :=
  ⟨source_device_selected dispatch e,source_torch_selected dispatch k⟩

#axiomsClean source_device_selected
#axiomsClean source_torch_selected
#axiomsClean source_profiles_selected
#specNonCircular BackendABI avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular DispatchLibrary avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating.Profiled

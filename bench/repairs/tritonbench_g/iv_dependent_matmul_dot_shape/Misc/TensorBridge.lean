import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Correct
import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.FPEquiv
import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Misc.HostExecution

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.TensorBridge
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- All five readable source quotations execute over arbitrary Real allocation
snapshots. The result is the exact allocated C Tensor with dense full-K values,
metadata, original input snapshots and the complete caller storage frame. -/
theorem matmul_checked_real {Device Error : Type} (rest : Algebra ℝ)
    (api : Wrapper.Calls ℝ Device Error) (r : Request) (before : Wrapper.Heap ℝ)
    (defaultFP32 cuda : Bool) (h : Contract.Preconditions r defaultFP32 cuda)
    (d : Device) (p : Wrapper.Prepared ℝ r before) (hp : Wrapper.PreparedBy api r before d p)
    (hl : CheckedHost.LaunchReady api d p) :
    VeriTile.Spec.Real (∃ out,
      HostExecution.call (RealExecution.backend rest) .source api r before defaultFP32 cuda=some out ∧
      CheckedHost.Post p out (Target.mathematical r.inner
        (Ideal.inputA p.request p.a.after.state) (Ideal.inputB p.request p.b.after.state))) := by
  have l := IvDependentMatmul.Contract.launchable p.request
    (CheckedHost.prepared_admitted p h.2.2.toAdmitted)
  obtain ⟨final,he,hm⟩ := Kernel.real_grid rest p.request Wrapper.layout
    p.realAllocation p.realAddressable l p.c.after.state
  have post := HostExecution.post_transfer p _ _
    (CheckedHost.real_post p h.2.2.toAdmitted .source) final hm
  exact ⟨HostExecution.returned p final,
    HostExecution.call_prepared _ _ api r before defaultFP32 cuda h d p hp hl final he,post⟩

#axiomsClean matmul_checked_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.TensorBridge

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.TensorBridge
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def backend (rest : Algebra Word) (profile : Target.NativeProfile) : Algebra Word :=
  FPValue.backend rest .ieeeValue
    (Instructions.dotPort (Instructions.referenceInstructions profile.reference))
    (Instructions.accumulatorPort (Instructions.referenceInstructions profile.reference))

def expected (profile : Target.NativeProfile) (r : Request) (aSnapshot bSnapshot : State Word) :=
  Target.native profile.reference r.inner r.blockInner (Target.inputA r aSnapshot) (Target.inputB r bSnapshot)

/-- The fixed emitted TF32 MMA lane is the only ISA arithmetic leaf. The
quoted native graph preserves its live FP32 carry, every tail operand and both
terminal conversions. Physical stores execute from those proved values. -/
theorem matmul_checked_native {Device Error : Type} (rest : Algebra Word)
    (profile : Target.NativeProfile) (api : Wrapper.Calls Word Device Error)
    (r : Request) (before : Wrapper.Heap Word) (defaultFP32 cuda : Bool)
    (h : Contract.Preconditions r defaultFP32 cuda) (d : Device)
    (p : Wrapper.Prepared Word r before) (hp : Wrapper.PreparedBy api r before d p)
    (hl : CheckedHost.LaunchReady api d p) (original prepared : OutputBytes.Read)
    (physical : ByteHost.Allocation p original prepared) :
    ∃ out bytes,
      HostExecution.call (backend rest profile) .accumulator api r before defaultFP32 cuda=some out ∧
      CheckedHost.Post p out (expected profile p.request p.a.after.state p.b.after.state) ∧
      (ByteExecution.grid
        (Execution.fp .ieeeValue
          (Instructions.dotPort (Instructions.referenceInstructions profile.reference))
          (Instructions.accumulatorPort (Instructions.referenceInstructions profile.reference)) p.request)
        .accumulator p.a.tensor.base p.b.tensor.base p.c.tensor.base
        (List.range (programs p.request)) (p.c.after.state,prepared)).2=bytes ∧
      ByteHost.Post p original prepared bytes (expected profile p.request p.a.after.state p.b.after.state) := by
  let R := profile.reference
  let D := Instructions.dotPort (Instructions.referenceInstructions R)
  let F := Instructions.accumulatorPort (Instructions.referenceInstructions R)
  let a := p.allocation p.c.after.state p.typedA p.typedB
  have addressable := p.addressable p.c.after.state p.typedA p.typedB
  have l := IvDependentMatmul.Contract.launchable p.request
    (CheckedHost.prepared_admitted p h.2.2.toAdmitted)
  obtain ⟨final,he,hm⟩ := Kernel.native_grid rest R.cfg D F p.request p.c.after.state a addressable l p.c.after.state
  have logical := HostExecution.target_transfer R p (HostExecution.returned p final)
    (HostExecution.post_transfer p _ _ (CheckedHost.fp_post rest R.cfg D F p h.2.2.toAdmitted .accumulator) final hm)
  have bytes := HostExecution.bytes_from_quotation rest R p h.2.2 prepared physical.outputC final hm
  exact ⟨HostExecution.returned p final, OutputBytes.render prepared (ByteHost.cView p) final,
    HostExecution.call_prepared _ _ api r before defaultFP32 cuda h d p hp hl final he,
    logical, bytes, ByteHost.rendered_post p original prepared physical _ _ logical⟩

#specNonCircular expected avoiding [HostExecution.call, Kernel.grid, Execution.program]
#axiomsClean matmul_checked_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.TensorBridge

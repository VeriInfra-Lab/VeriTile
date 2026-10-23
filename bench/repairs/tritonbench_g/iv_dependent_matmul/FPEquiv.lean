import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.ByteHost
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Instructions

namespace VeriTile.Bench.Repairs.IvDependentMatmul.FPEquiv
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Original A/B allocation snapshots, K8 TF32 instruction groups with live
FP32 carry, terminal FP16 narrowing, and FP32 widening. Source execution and
output memory are not inputs to this independent compiled recurrence. -/
def expected (R : Instructions.Reference) (r : Request) (inputs : State Word) : Nat → Nat → Word :=
  reference R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
    (Instructions.accumulatorPort (Instructions.referenceInstructions R)) .accumulator r inputs

/-- Complete checked native call. The only numerical premises are the four
fixed instruction/conversion correspondences; no split/fused equality or
variable-shape dot equality is assumed. Actual C pointers execute every byte
store, and readback, input preservation, allocation/return, and caller framing
are composed through all five schedules. -/
specification matmul_checked_if_local {Device Error : Type} (rest : Algebra Word)
    (I : Instructions.Native) (R : Instructions.Reference) (numeric : Instructions.Correspondence I R)
    (api : Wrapper.Calls Word Device Error) (r : Request) (before : Wrapper.Heap Word)
    (bytesFor : Wrapper.Prepared Word r before → Read) (original : Read) (defaultFP32 cuda : Bool)
    (h : Contract.Preconditions r defaultFP32 cuda) (d : Device) (p : Wrapper.Prepared Word r before)
    (hp : Wrapper.PreparedBy api r before d p) (hl : CheckedHost.LaunchReady api d p)
    (physical : ByteHost.Allocation p original (bytesFor p)) :
    ∃ out bytes,
      ByteHost.call (Instructions.arithmetic R.cfg I) .accumulator api r before bytesFor defaultFP32 cuda=some (out,bytes) ∧
      CheckedHost.Post p out (expected R p.request p.b.after.state) ∧
      ByteHost.Post p original (bytesFor p) bytes (expected R p.request p.b.after.state) := by
  have native := Instructions.native_eq I R numeric
  subst I
  have arithmetic : Instructions.arithmetic R.cfg (Instructions.referenceInstructions R)=
      Execution.fp R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
        (Instructions.accumulatorPort (Instructions.referenceInstructions R)) := by
    funext request
    exact Instructions.arithmetic_reference R request
  have logical := CheckedHost.fp_post rest R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
    (Instructions.accumulatorPort (Instructions.referenceInstructions R)) p h.2.2 .accumulator
  have called := ByteHost.call_prepared (Instructions.arithmetic R.cfg (Instructions.referenceInstructions R))
    .accumulator api r before bytesFor defaultFP32 cuda h d p hp hl
  rw [ByteHost.result_render _ _ p h.2.2 _ physical.outputC,arithmetic] at called
  exact ⟨_,_,called,logical,ByteHost.rendered_post p original (bytesFor p) physical _ _ logical⟩

#specNonCircular expected avoiding [ByteHost.call,CheckedHost.call,Execution.host,Execution.program]
#axiomsClean matmul_checked_if_local
end
end VeriTile.Bench.Repairs.IvDependentMatmul.FPEquiv

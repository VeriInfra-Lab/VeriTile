import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.NativePhases
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.RawNative
open VeriTile Triton
open VeriTile.Bench.Repairs.LightningAttention.Native (Instructions Plan LocalConformance)
noncomputable section

/-- Raw forward source/native execution, independent fixed-K8 native recurrence
and every untouched word. No Tensor object or application admission is used. -/
theorem forward_fp_equiv (d : Host.Dimensions) (input : Native.WordMemory)
    (actual selected : Instructions) (plan : Plan) (numeric : LocalConformance actual selected) :
    Native.forwardExecute actual plan d input=some (Native.forwardReference selected plan d input) ∧
      (∀ r o,¬Native.ForwardWritten d r o → Native.forwardReference selected plan d input r o=input r o) :=
  Native.forward_native actual selected plan d input numeric

/-- Both native backward stages consume evolving raw memory; their complete
source execution, independent recurrence and exterior frame are established. -/
theorem backward_fp_equiv (d : Host.Dimensions) (input : Native.WordMemory)
    (actual selected : Instructions) (plan : Plan) (numeric : LocalConformance actual selected) :
    Native.backwardExecute actual plan d input=some (Native.backwardReference selected plan d input) ∧
      (∀ r o,¬Native.BackwardWritten d r o → Native.backwardReference selected plan d input r o=input r o) :=
  Native.backward_native actual selected plan d input numeric

#axiomsClean forward_fp_equiv
#axiomsClean backward_fp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.RawNative

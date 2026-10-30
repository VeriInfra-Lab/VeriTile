import bench.tritonbench_g.attention_llama.Proofs.RealSourceTransport
import bench.tritonbench_g.attention_llama.Proofs.RealBacking

/-! Connection to the previously proved mathematical result. Its context can
already contain the runtime scalar binding: the old theorem permits arbitrary
initial registers. This avoids an additional executor or a reset-state rule. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealSourceTransport
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostResult
open RealScalarInvariant
noncomputable section

theorem literalSource_realConfig (c : Context) (Out : RegionName)
    (soz soh som son : Nat) (fp8 : Bool) :
    literalSource (Backing.realConfig c Out soz soh som son fp8) c.scale =
      Backing.realSource c Out soz soh som son fp8 := by
  cases fp8 <;> rfl

theorem backing_execution_eq (c : Context) (Out : RegionName)
    (soz soh som son : Nat) (fp8 : Bool) (s : BlockState) (bound : Bound c.scale s) :
    exec (Source.kernel (Backing.realConfig c Out soz soh som son fp8)).eraseDType.toAlgKernel s =
      exec (Backing.realSource c Out soz soh som son fp8).eraseDType.toAlgKernel s := by
  rw [execution_eq _ c.scale s bound,literalSource_realConfig]

theorem backing_traceSafe_iff (c : Context) (Out : RegionName)
    (soz soh som son : Nat) (fp8 : Bool) (bounds : RegionBounds)
    (s : BlockState) (bound : Bound c.scale s) :
    Kernel.TraceSafe bounds
      (Source.kernel (Backing.realConfig c Out soz soh som son fp8)).eraseDType.toAlgKernel s ↔
    Kernel.TraceSafe bounds (Backing.realSource c Out soz soh som son fp8).eraseDType.toAlgKernel s := by
  rw [traceSafe_iff _ c.scale bounds s bound,literalSource_realConfig]

theorem bound_contextFor (c : Context) (bound : Bound c.scale c.initial)
    (i : HostGrid.Index B heads c.H c.D) :
    Bound c.scale (contextFor c i).initial := bound

#axiomsClean backing_execution_eq
#axiomsClean backing_traceSafe_iff
#axiomsClean bound_contextFor
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealSourceTransport

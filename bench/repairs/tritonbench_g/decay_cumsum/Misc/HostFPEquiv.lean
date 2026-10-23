import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.decay_cumsum.Kernels
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Contract
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativePublic

/-! The public FP entry uses the source-specific native FMA graphs. The generic
structural theorem is only an internal step in that proof. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.NativeFPEquiv
open VeriTile.Triton VeriTile.Torch FP.Structural PackedStorage Native
noncomputable section

specification checked_whole_wrapper_equiv (compiler : Compiler) (instructions : Instructions)
    (h : Call) (kind : Kind) (observed : RegionName → TensorMetadata) (scaleIsFloat : Bool)
    (contract : InputContract h kind observed scaleIsFloat) (scale : Word)
    (api : WholeHost.API h kind Byte (Cell Node)) (targets : RegionName → Host.Target)
    (_binding : TensorBinding h kind observed targets)
    (before : Copyback.Heap Byte) (live : Nat → Prop)
    (ordinaryPreparation : WholeHost.PreparationLaws h kind api targets decode before live)
    (ordinaryCopy : WholeHost.CopyLaws api)
    (localCompilation : LocalCompilation compiler instructions h kind api) :
    Host.Outcome h kind targets encodeWord (reference instructions h kind scale targets before) before
      (Host.guarded h kind observed scaleIsFloat
        (run compiler h kind scale api targets before) before) := by
  rw [Host.guarded_eq contract]
  exact whole_wrapper_equiv compiler instructions h kind scale api targets before live
    ordinaryPreparation ordinaryCopy localCompilation

/-- The source PID and arange bounds follow from actual Python checks. -/
theorem checked_native_addresses (contract : InputContract h kind observed scaleIsFloat)
    (nonempty : 0<h.B*h.H*h.T*h.DK) (phase : Phase) (hp : phase∈phases h)
    (allocation : NativeIndex.Allocation h) (prog : Geometry.Program h phase)
    (step : Fin phase.span) (lane : Fin h.BK) :
    let pid := contract.pid nonempty phase hp prog
    (NativeIndex.offsetWord h phase pid step.val lane.val).toInt=
      (NativeIndex.relative h phase pid step.val lane.val:Int) ∧
    (BitVec.ofNat 64 (allocation.buffer.base+phase.start*h.width*allocation.buffer.bytes)+
      NativeIndex.offsetWord h phase pid step.val lane.val*BitVec.ofNat 64 allocation.buffer.bytes).toNat=
      allocation.buffer.base+Geometry.offset h phase prog step.val lane.val*allocation.buffer.bytes ∧
    allocation.buffer.base+Geometry.offset h phase prog step.val lane.val*allocation.buffer.bytes+
      allocation.buffer.bytes≤18446744073709551616 :=
  NativeABI.source_addresses h phase hp allocation contract.arange (contract.pid nonempty phase hp prog) step lane

#axiomsClean checked_whole_wrapper_equiv
#axiomsClean checked_native_addresses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.NativeFPEquiv

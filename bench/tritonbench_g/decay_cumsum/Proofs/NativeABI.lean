import bench.tritonbench_g.decay_cumsum.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.DecayCumsum.NativeABI
open Geometry NativeIndex

/-- Ordinary metadata of a successful `tl.arange(0,BK)` compilation in the
pinned Triton version. Its existing block-type constructor enforces at most
1048576 elements. This is a compiler/setup outcome, not a Python input filter.
Unsupported BK still has the wrapper's ordinary setup-failure outcome. -/
structure CompiledArange (h : Call) : Prop where
  blockLimit : h.BK≤1048576

theorem lane_signed32 (h : Call) (compiled : CompiledArange h) (lane : Fin h.BK) :
    lane.val<2147483648 := by have bound := compiled.blockLimit; have inside := lane.isLt; omega

/-- No kernel-index oracle: concrete widened source expressions are evaluated
and their allocated byte addresses proved from actual metadata and the PID ABI. -/
theorem source_addresses (h : Call) (p : Phase) (hp : p∈phases h)
    (allocation : Allocation h) (compiled : CompiledArange h) (prog : PID h p)
    (step : Fin p.span) (lane : Fin h.BK) :
    (offsetWord h p prog step.val lane.val).toInt=(NativeIndex.relative h p prog step.val lane.val:Int) ∧
    (BitVec.ofNat 64 (allocation.buffer.base+p.start*h.width*allocation.buffer.bytes)+
      offsetWord h p prog step.val lane.val*BitVec.ofNat 64 allocation.buffer.bytes).toNat=
      allocation.buffer.base+Geometry.offset h p prog.toProgram step.val lane.val*allocation.buffer.bytes ∧
    allocation.buffer.base+Geometry.offset h p prog.toProgram step.val lane.val*allocation.buffer.bytes+
      allocation.buffer.bytes≤18446744073709551616 :=
  NativeIndex.source_addresses h p hp allocation prog step lane (lane_signed32 h compiled lane)

#axiomsClean source_addresses
end VeriTile.Bench.Repairs.DecayCumsum.NativeABI

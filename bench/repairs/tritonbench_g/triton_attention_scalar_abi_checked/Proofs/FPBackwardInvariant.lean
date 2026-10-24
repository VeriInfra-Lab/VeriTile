import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardReference
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
def backwardByteBase (p : BackwardParameters) (s : State α) :=
  (s.pids 0/p.heads)*p.batchStride+(s.pids 0%p.heads)*p.headStride
structure BackwardContext (p : BackwardParameters) (initial current : State α) : Prop where
  undef : current.undef=initial.undef
  pids : current.pids=initial.pids
  hz : current.regs .nat [] "off_hz"=some (fun _ => initial.pids 0)
  z : current.regs .nat [] "off_z"=some (fun _ => initial.pids 0/p.heads)
  h : current.regs .nat [] "off_h"=some (fun _ => initial.pids 0%p.heads)
  strideZ : current.regs .nat [] "stride_qz_2d"=some (fun _ => p.batchStride/p.head)
  strideH : current.regs .nat [] "stride_qh_2d"=some (fun _ => p.headStride/p.head)
  dqBase : current.regs .ptr [] "DQ"=some (fun _ => (p.dq,backwardByteBase p initial))
structure BackwardOuterInvariant (p : BackwardParameters) (initial : State α)
    (c : Nat) (mem : Memory α) (current : State α) extends BackwardContext p initial current : Prop where
  memory : current.mem=mem
  keyPtr : current.regs .blockPtr [p.blockM,p.head] "k_tile_ptr"=some (fun _ => backwardPointer p initial p.k c)
  valuePtr : current.regs .blockPtr [p.blockM,p.head] "v_tile_ptr"=some (fun _ => backwardPointer p initial p.v c)
  dkPtr : current.regs .blockPtr [p.blockM,p.head] "dk_tile_ptr"=some (fun _ => backwardPointer p initial p.dk c)
  dvPtr : current.regs .blockPtr [p.blockM,p.head] "dv_tile_ptr"=some (fun _ => backwardPointer p initial p.dv c)
structure BackwardEntryInvariant (p : BackwardParameters) (initial : State α)
    (c : Nat) (mem : Memory α) (current : State α)
    extends BackwardOuterInvariant p initial c mem current : Prop where
  qPtr : current.regs .blockPtr [p.blockM,p.head] "q_tile_ptr"=some (fun _ => backwardPointer p initial p.q c)
  doPtr : current.regs .blockPtr [p.blockM,p.head] "do_tile_ptr"=some (fun _ => backwardPointer p initial p.gradient c)
  dqPtr : current.regs .blockPtr [p.blockM,p.head] "dq_tile_ptr"=some (fun _ => backwardPointer p initial p.dq c)
structure BackwardInnerInvariant (p : BackwardParameters) (initial : State α)
    (keyBlock queryBlock : Nat) (outerMem : Memory α) (st : BackwardInner α p) (current : State α)
    extends BackwardOuterInvariant p initial keyBlock st.mem current : Prop where
  lo : current.regs .nat [] "lo"=some (fun _ => keyBlock*p.blockM)
  rows : current.regs .nat [p.blockM] "offs_m"=some (fun i => i.1.val)
  cols : current.regs .nat [p.blockM] "offs_n"=some (fun i => keyBlock*p.blockM+i.1.val)
  qPtr : current.regs .blockPtr [p.blockM,p.head] "q_tile_ptr"=some (fun _ => backwardPointer p initial p.q queryBlock)
  doPtr : current.regs .blockPtr [p.blockM,p.head] "do_tile_ptr"=some (fun _ => backwardPointer p initial p.gradient queryBlock)
  dqPtr : current.regs .blockPtr [p.blockM,p.head] "dq_tile_ptr"=some (fun _ => backwardPointer p initial p.dq queryBlock)
  deltaPtr : current.regs .ptr [] "D_ptrs"=some (fun _ => (p.delta,initial.pids 0*(p.blockM*p.blocks)))
  maxPtr : current.regs .ptr [] "m_ptrs"=some (fun _ => (p.maximum,initial.pids 0*(p.blockM*p.blocks)))
  key : current.regs p.dtype.toTileDType [p.blockM,p.head] "k"=some (backwardRead p initial outerMem p.k keyBlock)
  value : current.regs p.dtype.toTileDType [p.blockM,p.head] "v"=some (backwardRead p initial outerMem p.v keyBlock)
  dqDeadPtrs : current.regs .ptr [p.blockM,p.head] "dq_ptrs"=some (fun i =>
    (p.dq,backwardByteBase p initial+(queryBlock*p.blockM+i.1.val)*p.head+i.2.1.val))
  dk : current.regs .real [p.blockM,p.head] "dk"=some st.dk
  dv : current.regs .real [p.blockM,p.head] "dv"=some st.dv
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source

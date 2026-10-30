import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPForwardReference
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
structure ForwardInvariant (P : Primitives α) (p : ForwardParameters) (initial : State α)
    (c : Nat) (st : ForwardAccumulator α p) (current : State α) : Prop where
  memory : current.mem=initial.mem
  undef : current.undef=initial.undef
  pids : current.pids=initial.pids
  start : current.regs .nat [] "start_m"=some (fun _ => initial.pids 0)
  hz : current.regs .nat [] "off_hz"=some (fun _ => initial.pids 1)
  rows : current.regs .nat [p.blockM] "offs_m"=some (fun i => forwardQueryRow p initial i.1)
  cols : current.regs .nat [p.blockN] "offs_n"=some (fun i => i.1.val)
  query : current.regs p.dtype.toTileDType [p.blockM,p.head] "q"=some (forwardQ p initial)
  keyPtr : current.regs .blockPtr [p.blockN,p.head] "k_tile_ptr"=some (fun _ => forwardKVPointer p initial p.k c)
  valuePtr : current.regs .blockPtr [p.blockN,p.head] "v_tile_ptr"=some (fun _ => forwardKVPointer p initial p.v c)
  outputPtr : current.regs .blockPtr [p.blockM,p.head] "out_tile_ptr"=some (fun _ => forwardOutPointer p initial)
  maximum : current.regs .real [p.blockM] "m_prev"=some st.maximum
  denominator : current.regs .real [p.blockM] "l_prev"=some st.denominator
  numerator : current.regs .real [p.blockM,p.head] "acc"=some st.numerator
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source

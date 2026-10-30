import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16Reference
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue (Primitives backend)
noncomputable section
variable {α : Type} [Inhabited α]
def qPointer (p : Parameters) (s : State α) : BlockPtr :=
  ⟨p.q,headBase p s,[p.length,p.head],[p.blockM,p.head],[p.qRowStride,p.qColStride],[s.pids 0*p.blockM,0]⟩
def kPointer (p : Parameters) (s : State α) (c : Nat) : BlockPtr :=
  ⟨p.k,kvBase p s,[p.head,p.length],[p.head,p.blockN],[p.kDimStride,p.kSeqStride],[0,c*p.blockN]⟩
def vPointer (p : Parameters) (s : State α) (c : Nat) : BlockPtr :=
  ⟨p.v,valueBase p s,[p.length,p.head],[p.blockN,p.head],[p.vSeqStride,p.vDimStride],[c*p.blockN,0]⟩
def outputPointer (p : Parameters) (s : State α) : BlockPtr :=
  ⟨p.output,outputBase p s,[p.length,p.head],[p.blockM,p.head],
    [p.outRowStride,p.outColStride],[s.pids 0*p.blockM,0]⟩

structure Invariant (P : Primitives α) (p : Parameters) (initial : State α)
    (c : Nat) (st : Accumulator α p) (current : State α) : Prop where
  memory : current.mem=initial.mem
  start : current.regs .nat [] "start_m"=some (fun _ => initial.pids 0)
  qoffset : current.regs .nat [] "q_offset"=some (fun _ => headBase p initial)
  ooffset : current.regs .nat [] "o_offset"=some (fun _ => outputBase p initial)
  boffset : current.regs .nat [] "b_offset"=some (fun _ => biasBase p initial)
  brows : current.regs .nat [p.blockM] "b_ptr_offsets_m"=some (fun i => i.1.val)
  bias : current.regs .bf16 [p.blockM,p.blockN] "b1"=some (b1Value P p initial)
  query : current.regs .bf16 [p.blockM,p.head] "q"=some (qScaled P p initial)
  keyPtr : current.regs .blockPtr [p.head,p.blockN] "K_block_ptr"=some (fun _ => kPointer p initial c)
  valuePtr : current.regs .blockPtr [p.blockN,p.head] "V_block_ptr"=some (fun _ => vPointer p initial c)
  maximum : current.regs .real [p.blockM] "m_i"=some st.maximum
  denominator : current.regs .real [p.blockM] "l_i"=some st.denominator
  numerator : current.regs .real [p.blockM,p.head] "acc"=some st.numerator
  lo : current.regs .nat [] "lo"=some (fun _ => 0)
  hi : current.regs .nat [] "hi"=some (fun _ => p.length)
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source

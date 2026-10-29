import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.FPReference
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def valuePointer {α : Type} (p : Parameters) (s : State α) : Values α .ptr [1,p.dimensions] :=
  fun i => (p.values,s.pids 1*p.valueHeadStride+i.2.1.val*p.valueDimStride)

structure Ready {α : Type} [Inhabited α] (p : Parameters) (s t : State α)
    (a : Accumulator α p.dimensions) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  batch : t.regs .nat [] "cur_batch" = some (fun _ => s.pids 0)
  head : t.regs .nat [] "cur_head" = some (fun _ => s.pids 1)
  seqlen : t.regs .nat [] "cur_batch_seq_len" = some (fun _ => sequenceLength p s)
  start : t.regs .nat [] "cur_batch_start_loc" = some (fun _ => sequenceStart p s)
  lanes : t.regs .nat [p.tokens] "offs_n" = some (fun i => i.1.val)
  dimensions : t.regs .nat [p.dimensions] "offs_d" = some (fun i => i.1.val)
  locbase : t.regs .nat [] "off_b_loc" = some (fun _ => locationBase p s)
  valueptr : t.regs .ptr [1,p.dimensions] "v_ptrs" = some (valuePointer p s)
  maximum : t.regs .real [] "e_max" = some (fun _ => a.maximum)
  normalizer : t.regs .real [] "e_sum" = some (fun _ => a.normalizer)
  output : t.regs .real [p.dimensions] "acc" = some a.output

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ final, run M (prelude p) s = some final ∧ Ready p s final (initialAccumulator M p) := by
  simp [prelude,source,sourceBody,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat]
  constructor <;> simp [State.setReg,sequenceLength,sequenceStart,locationBase,valuePointer,
    initialAccumulator,bop]
  all_goals first | rfl | (funext i; simp [bop,valuePointer,Region.cast])
#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue

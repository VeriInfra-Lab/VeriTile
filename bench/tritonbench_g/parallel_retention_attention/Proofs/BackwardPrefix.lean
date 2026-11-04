import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardLoop
import bench.tritonbench_g.parallel_retention_attention.Proofs.BackwardBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig BackwardAccumulator DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def backwardArgumentPrefix (c : BackwardConfig) := (backwardKernel c).surfaceBody.take 28
def backwardNumericalPrefix (c : BackwardConfig) := ((backwardKernel c).surfaceBody.drop 28).take 15
def backwardPrefix (c : BackwardConfig) := (backwardKernel c).surfaceBody.take 43

def backwardInitial {α : Type} (M : Algebra α) (c : BackwardConfig) : BackwardAccumulator α c.BTL c.BK c.BV :=
  ⟨fun _=>M.literal (some .fp32) .real 0,fun _=>M.literal (some .fp32) .real 0⟩

structure BackwardArgumentReady {α : Type} (c : BackwardConfig) (s t : State α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  argument0 : t.regs .int [] "__veritile_argument_16973089353068028100_0"=some (fun _=>(c.i_bh : Int))
  argument1 : t.regs .int [] "__veritile_argument_16973089353068028100_1"=some (fun _=>(c.i_c : Int))
  argument2 : t.regs .int [] "__veritile_argument_16973089353068028100_2"=some (fun _=>(c.i_k : Int))
  argument3 : t.regs .int [] "__veritile_argument_16973089353068028100_3"=some (fun _=>(c.i_v : Int))
  argument4 : t.regs .int [] "__veritile_argument_16973089353068028100_4"=some (fun _=>(c.i_h : Int))
  argument5 : t.regs .int [] "__veritile_argument_16973089353068028100_5"=some (fun _=>(c.s_qk_h : Int))
  argument6 : t.regs .int [] "__veritile_argument_16973089353068028100_6"=some (fun _=>(c.s_qk_t : Int))
  argument7 : t.regs .int [] "__veritile_argument_16973089353068028100_7"=some (fun _=>(c.s_qk_d : Int))
  argument8 : t.regs .int [] "__veritile_argument_16973089353068028100_8"=some (fun _=>(c.s_vo_h : Int))
  argument9 : t.regs .int [] "__veritile_argument_16973089353068028100_9"=some (fun _=>(c.s_vo_t : Int))
  argument10 : t.regs .int [] "__veritile_argument_16973089353068028100_10"=some (fun _=>(c.s_vo_d : Int))
  argument11 : t.regs .int [] "__veritile_argument_16973089353068028100_11"=some (fun _=>(c.B : Int))
  argument12 : t.regs .int [] "__veritile_argument_16973089353068028100_12"=some (fun _=>(c.H : Int))
  argument13 : t.regs .int [] "__veritile_argument_16973089353068028100_13"=some (fun _=>(c.T : Int))

theorem backward_arguments_run {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s : State α) :
    ∃ t,run M (backwardArgumentPrefix c) s=some t ∧ BackwardArgumentReady c s t := by
  simp [backwardArgumentPrefix,backwardKernel,backwardSource,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop]
  constructor <;> simp

theorem backward_numerical_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (h : BackwardArgumentReady c s t)
    (bounds : HelperIndexBounds c) :
    ∃ u,run (withDot M D) (backwardNumericalPrefix c) t=some u ∧
      BackwardReady M c s u (backwardInitial M c) := by
  simp [backwardNumericalPrefix,backwardKernel,backwardSource,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,ComputeDType.eraseDType,boundaryZero,h.mem,h.pids,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,
    bounds.query_cast,bounds.key_cast,bounds.value_cast,Index32.natural_product,
    evalOffsets,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,TileShape.indexToList,
    ofFloat,toFloat,withDot]
  constructor <;> try simp [h.mem,h.pids,h.undef,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,
    backwardKey,backwardValue,matrixRead,backwardInitial,decayedKey,bop,numeric,ofFloat,toFloat,withDot,
    logDecay,rowDecay,blockDecay,bounds.query_cast,bounds.key_cast,bounds.value_cast,
    Index32.natural_product]
  all_goals rfl

theorem backward_prefix_split (c : BackwardConfig) :
    backwardPrefix c=backwardArgumentPrefix c++backwardNumericalPrefix c:=rfl

theorem backward_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) (bounds : HelperIndexBounds c) :
    ∃ u,run (withDot M D) (backwardPrefix c) s=some u ∧
      BackwardReady M c s u (backwardInitial M c) := by
  obtain ⟨captured,hcapture,hargs⟩:=backward_arguments_run (withDot M D) c s
  obtain ⟨final,hnumeric,ready⟩:=backward_numerical_run M D c s captured hargs bounds
  refine ⟨final,?_,ready⟩
  rw [backward_prefix_split,run_append,hcapture,Option.bind_some,hnumeric]

#axiomsClean backward_arguments_run
#axiomsClean backward_numerical_run
#axiomsClean backward_prefix_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages

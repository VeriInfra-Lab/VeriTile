import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryLoop
import bench.tritonbench_g.parallel_retention_attention.Proofs.HelperBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig DotPrimitive withDot)
open FPStages (logDecay rowDecay blockDecay)
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def argumentPrefix (c : BackwardConfig) (dq : RegionName) := (source c dq).surfaceBody.take 28
def numericalPrefix (c : BackwardConfig) (dq : RegionName) := ((source c dq).surfaceBody.drop 28).take 8
def prefixCode (c : BackwardConfig) (dq : RegionName) := (source c dq).surfaceBody.take 36

structure ArgumentReady {α : Type} (c : BackwardConfig) (s t : State α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  argument0 : t.regs .int [] "__veritile_argument_18379874528259158415_0"=some (fun _=>(c.i_bh : Int))
  argument1 : t.regs .int [] "__veritile_argument_18379874528259158415_1"=some (fun _=>(c.i_c : Int))
  argument2 : t.regs .int [] "__veritile_argument_18379874528259158415_2"=some (fun _=>(c.i_k : Int))
  argument3 : t.regs .int [] "__veritile_argument_18379874528259158415_3"=some (fun _=>(c.i_v : Int))
  argument4 : t.regs .int [] "__veritile_argument_18379874528259158415_4"=some (fun _=>(c.i_h : Int))
  argument5 : t.regs .int [] "__veritile_argument_18379874528259158415_5"=some (fun _=>(c.s_qk_h : Int))
  argument6 : t.regs .int [] "__veritile_argument_18379874528259158415_6"=some (fun _=>(c.s_qk_t : Int))
  argument7 : t.regs .int [] "__veritile_argument_18379874528259158415_7"=some (fun _=>(c.s_qk_d : Int))
  argument8 : t.regs .int [] "__veritile_argument_18379874528259158415_8"=some (fun _=>(c.s_vo_h : Int))
  argument9 : t.regs .int [] "__veritile_argument_18379874528259158415_9"=some (fun _=>(c.s_vo_t : Int))
  argument10 : t.regs .int [] "__veritile_argument_18379874528259158415_10"=some (fun _=>(c.s_vo_d : Int))
  argument11 : t.regs .int [] "__veritile_argument_18379874528259158415_11"=some (fun _=>(c.B : Int))
  argument12 : t.regs .int [] "__veritile_argument_18379874528259158415_12"=some (fun _=>(c.H : Int))
  argument13 : t.regs .int [] "__veritile_argument_18379874528259158415_13"=some (fun _=>(c.T : Int))

  outerHead : t.regs .int [] "i_bh"=s.regs .int [] "i_bh"
  outerChunk : t.regs .int [] "i_c"=s.regs .int [] "i_c"
  outerKey : t.regs .int [] "i_k"=s.regs .int [] "i_k"
  outerValue : t.regs .int [] "i_v"=s.regs .int [] "i_v"
  outerDecay : t.regs .int [] "i_h"=s.regs .int [] "i_h"

theorem arguments_run {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (dq : RegionName) (s : State α) :
    ∃ t,run M (argumentPrefix c dq) s=some t ∧ ArgumentReady c s t := by
  simp [argumentPrefix,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop]
  constructor <;> simp

theorem numerical_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s t : State α) (h : ArgumentReady c s t)
    (bounds : HelperIndexBounds c) :
    ∃ u,run (withDot M D) (numericalPrefix c dq) t=some u ∧
      Ready M c s u 0 (fun _=>M.literal (some .fp32) .real 0) := by
  simp [numericalPrefix,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,h.mem,h.pids,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay,
    bounds.query_cast,bounds.key_cast,bounds.value_cast,Index32.natural_product,
    evalOffsets,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,TileShape.indexToList,
    ofFloat,toFloat,withDot]
  constructor <;> simp [h.mem,h.pids,h.undef,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay,
    outputGradient,matrixRead,keyPointer,valuePointer,bop,numeric,ofFloat,toFloat,withDot,
    logDecay,rowDecay,blockDecay,bounds.query_cast,bounds.key_cast,bounds.value_cast,
    Index32.natural_product]
  all_goals rfl

theorem prefix_split (c : BackwardConfig) (dq : RegionName) :
    prefixCode c dq=argumentPrefix c dq++numericalPrefix c dq:=rfl

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) (bounds : HelperIndexBounds c) :
    ∃ u,run (withDot M D) (prefixCode c dq) s=some u ∧
      Ready M c s u 0 (fun _=>M.literal (some .fp32) .real 0) := by
  obtain ⟨captured,hcapture,hargs⟩:=arguments_run (withDot M D) c dq s
  obtain ⟨final,hnumeric,ready⟩:=numerical_run M D c dq s captured hargs bounds
  refine ⟨final,?_,ready⟩
  rw [prefix_split,run_append,hcapture,Option.bind_some,hnumeric]

#axiomsClean arguments_run
#axiomsClean numerical_run
#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery

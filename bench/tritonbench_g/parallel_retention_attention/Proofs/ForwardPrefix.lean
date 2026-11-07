import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardLoop
import bench.tritonbench_g.parallel_retention_attention.Proofs.IndexBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (ForwardConfig DotPrimitive withDot queryChunk keyChunk valueChunk)
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def argumentPrefix (c : ForwardConfig) := (forwardKernel c).surfaceBody.take 18
def decodedPrefix (c : ForwardConfig) := ((forwardKernel c).surfaceBody.drop 18).take 5
def numericalPrefix (c : ForwardConfig) := ((forwardKernel c).surfaceBody.drop 23).take 9

structure ArgumentReady {α : Type} (c : ForwardConfig) (s t : State α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  argument0 : t.regs .int [] "__veritile_argument_12546945468679417523_0"=some (fun _=>(c.s_qk_h : Int))
  argument1 : t.regs .int [] "__veritile_argument_12546945468679417523_1"=some (fun _=>(c.s_qk_t : Int))
  argument2 : t.regs .int [] "__veritile_argument_12546945468679417523_2"=some (fun _=>(c.s_qk_d : Int))
  argument3 : t.regs .int [] "__veritile_argument_12546945468679417523_3"=some (fun _=>(c.s_vo_h : Int))
  argument4 : t.regs .int [] "__veritile_argument_12546945468679417523_4"=some (fun _=>(c.s_vo_t : Int))
  argument5 : t.regs .int [] "__veritile_argument_12546945468679417523_5"=some (fun _=>(c.s_vo_d : Int))
  argument6 : t.regs .int [] "__veritile_argument_12546945468679417523_6"=some (fun _=>(c.B : Int))
  argument7 : t.regs .int [] "__veritile_argument_12546945468679417523_7"=some (fun _=>(c.H : Int))
  argument8 : t.regs .int [] "__veritile_argument_12546945468679417523_8"=some (fun _=>(c.T : Int))

structure DecodedReady {α : Type} (c : ForwardConfig) (s t : State α) : Prop extends ArgumentReady c s t where
  chunk : t.regs .int [] "i_c"=some (fun _=>(s.pids 1 : Int))
  head : t.regs .int [] "i_bh"=some (fun _=>(s.pids 2 : Int))
  keyBlock : t.regs .int [] "i_k"=some (fun _=>((s.pids 0/((c.V+c.BV-1)/c.BV) : Nat) : Int))
  valueBlock : t.regs .int [] "i_v"=some (fun _=>((s.pids 0%((c.V+c.BV-1)/c.BV) : Nat) : Int))
  decayHead : t.regs .int [] "i_h"=some (fun _=>((s.pids 2%c.H : Nat) : Int))

theorem arguments_run {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig) (s : State α) :
    ∃ t,run M (argumentPrefix c) s=some t ∧ ArgumentReady c s t := by
  simp [argumentPrefix,forwardKernel,forwardSource,forwardPastedKernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop]
  constructor <;> simp

#axiomsClean arguments_run

theorem decoded_run {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s t : State α) (h : ArgumentReady c s t) :
    ∃ u,run M (decodedPrefix c) t=some u ∧ DecodedReady c s u := by
  simp [decodedPrefix,forwardKernel,forwardSource,forwardPastedKernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,numericLt,natLt,
    integralFloorDiv,integralMod,IntegerExpr.sameShape,h.pids,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,
    Index32.nat_remainder]
  constructor
  · constructor <;> simp [h.mem,h.pids,h.undef,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8]
  all_goals simp [State.setReg,bop,numeric,Index32.nat_remainder]
  all_goals funext i
  all_goals dsimp only [bop]
  all_goals simp [←Int.emod_def,Int.mul_comm]


#axiomsClean arguments_run
#axiomsClean decoded_run

def firstPrefix (c : ForwardConfig) := (forwardKernel c).surfaceBody.take 32
def initialAccumulator {α : Type} (M : Algebra α) (c : ForwardConfig) : TileIndex [c.BTL,c.BV] → α :=
  fun _=>M.literal (some .fp32) .real 0

theorem numerical_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s t : State α) (h : DecodedReady c s t) (bounds : ForwardIndexBounds c s) :
    ∃ u,run (withDot M D) (numericalPrefix c) t=some u ∧
      ForwardReady M c s u 0 (initialAccumulator M c) := by
  simp [numericalPrefix,forwardKernel,forwardSource,forwardPastedKernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,numericLt,natLt,
    integralFloorDiv,integralMod,IntegerExpr.sameShape,h.mem,h.pids,h.head,h.chunk,h.keyBlock,
    h.valueBlock,h.decayHead,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,bounds.query_cast,bounds.key_cast,bounds.value_cast,bounds.key_cast_eval,bounds.value_cast_eval,
    Index32.natural_product,Index32.nat_remainder,evalOffsets,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,TileShape.indexToList,ofFloat,toFloat,withDot]
  constructor <;> simp [h.mem,h.pids,h.undef,h.head,h.chunk,h.keyBlock,h.valueBlock,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,
    forwardQuery,initialAccumulator,matrixRead,keyPointer,valuePointer,queryChunk,keyChunk,
    valueChunk,bop,numeric,ofFloat,toFloat,withDot,logDecay,rowDecay,
    bounds.query_cast,bounds.key_cast,bounds.value_cast,bounds.key_cast_eval,bounds.value_cast_eval,Index32.natural_product]
  all_goals first | rfl | (funext i; simp [bop])

theorem prefix_split (c : ForwardConfig) :
    firstPrefix c=argumentPrefix c++decodedPrefix c++numericalPrefix c:=rfl

theorem first_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) (bounds : ForwardIndexBounds c s) :
    ∃ final,run (withDot M D) (firstPrefix c) s=some final ∧
      ForwardReady M c s final 0 (initialAccumulator M c) := by
  obtain ⟨captured,hcapture,hargs⟩:=arguments_run (withDot M D) c s
  obtain ⟨decoded,hdecode,hdecoded⟩:=decoded_run (withDot M D) c s captured hargs
  obtain ⟨final,hnumeric,ready⟩:=numerical_run M D c s decoded hdecoded bounds
  refine ⟨final,?_,ready⟩
  rw [prefix_split,run_append,run_append,hcapture,Option.bind_some,hdecode,Option.bind_some,hnumeric]

#axiomsClean numerical_run
#axiomsClean first_prefix_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages

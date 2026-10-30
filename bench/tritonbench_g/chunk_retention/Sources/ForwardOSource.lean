import bench.tritonbench_g.chunk_retention.Sources.KernelModel

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (c : Shape) (scale : ℝ) :=
  Kernels.forwardO "q" "k" "v" "h" "o"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    scale c.H c.T c.K c.V 64 c.BK c.BV
def parts (c : Shape) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel c scale).surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
abbrev Index (c : Shape) := TileIndex [64,c.BV]
abbrev ScoreIndex := TileIndex [64,64]
def row (s : State α) (i : Nat) := s.pids 1*64+i
def column (c : Shape) (s : State α) (j : Nat) := s.pids 0*c.BV+j
def feature (c : Shape) (f e : Nat) := f*c.BK+e
def rowDecay (M : Algebra α) (c : Shape) (s : State α) : TileIndex [64] → α :=
  fun i => KernelModel.decay M c (s.pids 2) ((i.1.val:Int)+1)
def scoreDecay (M : Algebra α) (c : Shape) (s : State α) : ScoreIndex → α :=
  fun i => if i.2.1.val ≤ i.1.val then
    KernelModel.decay M c (s.pids 2) ((i.1.val:Int)-i.2.1.val) else zero M

structure Context (M : Algebra α) (c : Shape) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  iv : current.regs .nat [] "i_v"=some (fun _ => initial.pids 0)
  it : current.regs .nat [] "i_t"=some (fun _ => initial.pids 1)
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  di : current.regs .real [64] "d_i"=some (rowDecay M c initial)
  ds : current.regs .real [64,64] "d_s"=some (scoreDecay M c initial)
def HasAcc (c : Shape) (a : Index c → α) (b : ScoreIndex → α) (s : State α) : Prop :=
  s.regs .real [64,c.BV] "b_o"=some a ∧ s.regs .real [64,64] "b_s"=some b

def entryExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    {t : State α // run (engine (withDot M D)) (parts c scale).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,Kernels.forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,
    natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem entryContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    Context M c s (entryExecution M D c scale s).val ∧
      HasAcc c (fun _ => zero M) (fun _ => zero M) (entryExecution M D c scale s).val := by
  constructor
  · constructor <;> simp [entryExecution,rowDecay,scoreDecay,KernelModel.decay,rate,zero,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Function.comp_def]
    all_goals norm_num [rowDecay,scoreDecay,KernelModel.decay,rate,zero]
    all_goals rfl
  · simp [HasAcc,entryExecution,zero]

def bodyExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Index c → α) (b : ScoreIndex → α)
    (ctx : Context M c initial current) (ha : HasAcc c a b current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => f))=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,Kernels.forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,
    natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.iv,ctx.it,ctx.bh,ctx.di,ctx.ds,ha.1,ha.2]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource

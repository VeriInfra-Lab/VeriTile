import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPOutput
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.ProducerGeometry
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPInter
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open ProducerGeometry (Shape)
open FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def kernel (c : Shape) (scale : ℝ) := SourceBridge.interNew (ProducerGeometry.layout c) ProducerGeometry.regions scale
def outer (c : Shape) (scale : ℝ) : Source.Conditional :=
  (Source.conditional? (kernel c scale).surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def inner (c : Shape) (scale : ℝ) : Source.Conditional :=
  (Source.conditional? (outer c scale).yes (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def parts (c : Shape) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (inner c scale).yes "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
abbrev Index := TileIndex [16,16]
variable {α : Type} [Inhabited α]

def rowStart (c : Shape) (s : State α) := s.pids 0*c.BT+s.pids 1/c.NC*16
def columnStart (c : Shape) (s : State α) := s.pids 0*c.BT+s.pids 1%c.NC*16
def row (c : Shape) (s : State α) (i : Nat) := rowStart c s+i
def column (c : Shape) (s : State α) (j : Nat) := s.pids 1%c.NC*16+j
def feature (c : Shape) (f e : Nat) := f*c.BK+e
def zero (M : Algebra α) := M.literal (some .fp32) .real 0

structure Context (c : Shape) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  it : current.regs .nat [] "i_t"=some (fun _ => initial.pids 0)
  ic : current.regs .nat [] "i_c"=some (fun _ => initial.pids 1)
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  ii : current.regs .nat [] "i_i"=some (fun _ => initial.pids 1/c.NC)
  ij : current.regs .nat [] "i_j"=some (fun _ => initial.pids 1%c.NC)
def HasAcc (a : Index → α) (s : State α) : Prop := s.regs .real [16,16] "b_A"=some a

def entryExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    {t : State α // run (engine (withDot M D)) ((outer c scale).before++(parts c scale).before) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,ProducerGeometry.layout,ProducerGeometry.regions,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem entryContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    Context c s (entryExecution M D c scale s).val ∧
      HasAcc (fun _ => zero M) (entryExecution M D c scale s).val := by
  constructor
  · constructor <;> simp [entryExecution] <;> rfl
  · simp [HasAcc,entryExecution,zero]

def bodyExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Index → α)
    (ctx : Context c initial current) (ha : HasAcc a current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => f))=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,ProducerGeometry.layout,ProducerGeometry.regions,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.it,ctx.ic,ctx.bh,ctx.ii,ctx.ij,ha]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPInter

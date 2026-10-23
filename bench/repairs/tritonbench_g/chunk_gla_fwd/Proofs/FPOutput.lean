import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.OutputGrid
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPMerge
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

/-- One total matrix primitive, with every dtype, precision and dot option
retained. This port imposes no numerical law and computes no kernel result. -/
structure DotPrimitive (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) →
    Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def withDot {α : Type} (M : Algebra α) (D : DotPrimitive α) : Algebra α :=
  { M with
    dot := fun p {_} {_} {_} {_} a b => some (D.apply p .real .real .real {} a b)
    dotWithOptions := fun p o {_} {_} {_} {_} a b => some (D.apply p .real .real .real o a b)
    dotTyped := fun p l r o options {_} {_} {_} {_} a b => some (D.apply p l r o options a b)}

def kernel (c : OutputGrid.Shape) (scale : ℝ) :=
  SourceBridge.outputNew (OutputGrid.layout c) OutputGrid.regions scale
def parts (c : OutputGrid.Shape) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel c scale).surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
abbrev Index (c : OutputGrid.Shape) := TileIndex [c.BT,c.BV]
variable {α : Type} [Inhabited α]

def row (c : OutputGrid.Shape) (s : State α) (i : Nat) := s.pids 1*c.BT+i
def column (c : OutputGrid.Shape) (s : State α) (j : Nat) := s.pids 0*c.BV+j
def feature (c : OutputGrid.Shape) (f e : Nat) := f*c.BK+e
def zero (M : Algebra α) := M.literal (some .fp32) .real 0

structure Context (c : OutputGrid.Shape) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  iv : current.regs .nat [] "i_v"=some (fun _ => initial.pids 0)
  it : current.regs .nat [] "i_t"=some (fun _ => initial.pids 1)
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  causal : current.regs .bool [c.BT,c.BT] "m_s"=some (fun i => decide (i.2.1.val ≤ i.1.val))
def HasAcc (c : OutputGrid.Shape) (a : Index c → α) (s : State α) : Prop :=
  s.regs .real [c.BT,c.BV] "b_o"=some a

def entryExecution (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) :
    {t : State α // run (engine (withDot M D)) (parts c scale).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,SourceBridge.outputNew,TypedOutput.original,TypedOutput.source,OutputGrid.layout,OutputGrid.regions,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem entryContext (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) :
    Context c s (entryExecution M D c scale s).val ∧
      HasAcc c (fun _ => zero M) (entryExecution M D c scale s).val := by
  constructor
  · constructor <;> simp [entryExecution] <;> rfl
  · simp [HasAcc,entryExecution,zero]

def bodyExecution (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Index c → α)
    (ctx : Context c initial current) (ha : HasAcc c a current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => f))=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,SourceBridge.outputNew,TypedOutput.original,TypedOutput.source,OutputGrid.layout,OutputGrid.regions,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.iv,ctx.it,ctx.bh,ctx.causal,ha]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput

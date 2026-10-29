import bench.repairs.tritonbench_g.chunk_retention_ops.Kernels
import bench.repairs.tritonbench_g.chunk_linear_attn.Model
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
abbrev Parameters := VeriTile.Bench.Repairs.ChunkLinearAttn.Parameters
abbrev Memory := VeriTile.Bench.Repairs.ChunkLinearAttn.Memory
abbrev PID := VeriTile.Bench.Repairs.ChunkLinearAttn.PID
abbrev Matrix := VeriTile.Bench.Repairs.ChunkLinearAttn.Matrix
variable {α : Type} [Inhabited α]

def kernel (p : Parameters) (scale : ℝ) (terminal initial : Bool) : ComputeKernel :=
  Source.backwardH "q" "incoming" "dhistory" "terminal" "initialGradient"
    (p.T*p.K) p.K 1 (p.T*p.V) p.V 1 (p.NT*p.K*p.V) p.V
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT terminal initial

def parts (p : Parameters) (scale : ℝ) (terminal initial : Bool) : Triton.Source.DynamicRange :=
  (Triton.Source.dynamicRange? (kernel p scale terminal initial).surfaceBody "j").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

/-- Executes the actual selected quotation's complete prefix, including the
signed head index, transcendental decay and optional terminal-state load. -/
def entry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (s : State α) :
    {t : State α // run (engine (withDot M D)) (parts p scale terminal initial).before s=some t} := by
  cases terminal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,Source.backwardH,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
    rfl

theorem entry_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (s : State α) :
    (entry M D p scale terminal initial s).val.mem=s.mem := by
  cases terminal <;> simp [entry]

#axiomsClean entry_memory
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward

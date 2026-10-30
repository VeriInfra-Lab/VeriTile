import bench.repairs.tritonbench_g.f8_conversion_utils.Kernels
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- The actual destination conversion, with source storage precision retained.
The legacy algorithm tag .real in the FP32 input arm is an opaque FP32 channel
under its default production profile; it is never interpreted as a Real here. -/
def cast {α : Type} (M : Algebra α) (p : Profile) (x : α) : α :=
  M.cast none p.input p.output (if p=Profile.encodeSingleE5 ∨ p=Profile.encodeSingleE4 then M.fp32Load x else x)

def stores {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B pid : Nat) (snapshot : State α) (s : State α) : State α :=
  (TileShape.allIndices [B]).foldl (fun current i =>
    if pid*B+i.1.val<N then
      current.write Y (pid*B+i.1.val)
        (.mk p.output.toTileDType (ofFloat p.output
          (cast M p (toFloat p.input ((snapshot.mem X (pid*B+i.1.val)).read p.input.toTileDType)))))
    else current) s

def memory {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B : Nat) (s : State α) :=
  let first := stores M p Y X N B (s.pids 0) s s
  if p.twice then (stores M p Y X N B (s.pids 0) s first).mem else first.mem

/-- Successful execution of each whole original default-E5 JIT. Both decode
stores use the same initially loaded register; no cast-idempotence is used. -/
theorem executes {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B : Nat) (s : State α) :
    ∃ final,FP.Structural.exec M (kernel p Y X N B) s=some final ∧
      final.mem=memory M p Y X N B s := by
  cases Y
  cases X
  cases p
  all_goals
    simp +contextual [kernel,toHalf,toFP8,
      VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource.toHalf,
      VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource.toFP8,
      FP.Structural.exec,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,natLt,bop,FP.Structural.store,Region.cast,ofFloat,toFloat,
      memory,stores,cast,Profile.input,Profile.output,Profile.twice,Option.bind,
      VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits.fold_writes_regs,
      VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits.fold_writes_mem]

#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source

import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairIO
open VeriTile Triton FP.Structural ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
noncomputable section

def invocation {α : Type} (M : Algebra α)
    (D : VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.DotPrimitive α)
    (p : Parameters) (pid : PID) : Invocation α :=
  { Interface.invocation M D p pid with kernel := ChunkDeltaFwdStaticDot.addressFold p }

def io (I : WordPrimitives) (p : Parameters) (schedule : List PID) : ProgramIO (BitVec 32) Expr :=
  Interface.io Native.algebra Native.dot p schedule Expr.bits (Reference.nativeObserve I)

def candidate (I : WordPrimitives) (p : Parameters) (schedule : List PID) : ProgramIO.Variant (io I p schedule) where
  schedule := fun _ => schedule.map (invocation Native.algebra Native.dot p)
  observe := (io I p schedule).observe

/-- Original full selected source grid, all FP32 words and unchanged native ISA. -/
def original (I : WordPrimitives) (p : Parameters) (schedule : List PID) :=
  ProgramIO.Implementation.original (io I p schedule) (fun _ => True)

/-- Full candidate grid; only compile-time state/history address products differ. -/
def addressFold (I : WordPrimitives) (p : Parameters) (schedule : List PID) :=
  ProgramIO.Implementation.ofVariant (candidate I p schedule) (fun _ => True)

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairIO

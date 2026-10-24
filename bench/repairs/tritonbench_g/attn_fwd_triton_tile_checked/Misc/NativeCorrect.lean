import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.NativeSchedule
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
open VeriTile Triton FP.Structural
open AttnFwdTriton.TypedExecution Graph
noncomputable section

/-- Full selected source grid: FP32 input/scales, TF32 score dot, FP16 value
dot and BF16 output. Existing physical Views imply geometric separation;
the checked memory-equivalence bridge preserves the original grid. -/
def nativeIO (p : Config) : ProgramIO Expr Expr := NativeInterface.programIO p

def nativeObserver (ops : Operations) (p : Config)
    (_o : Fin (nativeIO p).outputs.length) (_j : TileIndex (nativeIO p).outputs[_o].shape) :
    Expr → Word := evaluate ops

/-- Existing local instruction meanings observe the independent source-order
recurrence. Every raw FP32 word remains covered without a new numerical law. -/
def nativeFormula (ops : Operations) (p : Config) (values : NativeInterface.Inputs p) :
    (nativeIO p).ObservedOutputs Word :=
  (nativeIO p).mapOutputs (nativeObserver ops p) (NativeInterface.formula p values)

theorem whole_native (ops : Operations) (p : Config) (stage : p.stage=3)
    (nonoverlap : NonOverlapping p) (separate : ReadLocality.Separated p) :
    (nativeIO p).Observes (nativeObserver ops p) (nativeFormula ops p) :=
  ProgramIO.Observes.of_implements (NativeInterface.implements p stage nonoverlap separate)
    (nativeObserver ops p)

-- Operations are primitive model boundaries; this is source-to-native
-- observation, not an optimization equivalence with new numerical laws.
#axiomsClean whole_native
#specNonCircular nativeFormula avoiding [FP.Structural.exec,FP.Structural.run,runSchedule,
  AttnFwdTriton.TypedExecution.reference]
#stmtSurfaceSubset whole_native ⊆ [ProgramIO.Observes,nativeIO,nativeFormula,
  nativeObserver,Operations,Config,Config.stage,NonOverlapping,ReadLocality.Separated,
  Expr,Word,Graph.instInhabitedExpr]
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked

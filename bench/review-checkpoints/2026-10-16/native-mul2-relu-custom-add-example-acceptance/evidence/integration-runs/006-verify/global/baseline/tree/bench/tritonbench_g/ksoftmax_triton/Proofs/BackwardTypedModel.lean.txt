import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSource
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- An inhabited exact-real backend for the actual typed interpreter. This
witness supplies no IEEE/native admission claim. Every operation used by
the source has its usual mathematical meaning. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ op a => match op with
    | .exp => WithBot.realExp a
    | _ => some 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v =>
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data
  reduceSumTyped := fun _ d {_} ax keep v => some (fun i =>
    ofFloat d ((Tile.reduceSum ax keep ⟨fun j => toFloat d (v j)⟩).data i))

def channel : Typed.Storage → FloatDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5

def initial (storage : Typed.Storage) (log : Bool) : State (WithBot ℝ) where
  mem := fun r _ => .mk (channel storage).toTileDType
    (ofFloat (channel storage)
      (some (if r = "G" then 2 else if r = "O" then (if log then 0 else 1 / 4) else 7)))
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 2 else if ax = 1 then 1 else 0
  numPids := fun _ => 3
  undef := fun dtype _ _ => defaultValue dtype

end
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution


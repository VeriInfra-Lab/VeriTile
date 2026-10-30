import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSource
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps

/-! Temporary forward witness model. No IEEE or native admission is claimed.
It is separate from the frozen backward model: LOG and typed maximum have
their actual extended-real meanings here, including negative infinity.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical

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
    | .log => WithBot.realLog a
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
  reduceMaxTyped := fun _ d {_} ax keep v =>
    (Tile.reduceMax ax keep ⟨fun j => toFloat d (v j)⟩).map
      (fun t i => ofFloat d (t.data i))

/-- pids=(2,1), X row starts at 39, qk mask at 13, bk mask at 26,
and output at 29. The active qk samples are 5 and 6; bk samples are
26 and 28. Their unequal differences distinguish both a wrong storage
channel (two default zeroes) and the wrong qk/bk address layout.
All initial cells use the actual fp16 channel and all registers are empty. -/
def initial : State (WithBot ℝ) where
  mem := fun r o => Cell.mk .fp16 (some (if r = "X" then (0 : ℝ)
    else if r = "M" then
      (if o = 13 then 5 else if o = 14 then 6
       else if o = 26 then 26 else if o = 27 then 28 else 0) else 7))
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 2 else if ax = 1 then 1 else 0
  numPids := fun _ => 3
  undef := fun dtype _ _ => defaultValue dtype

theorem initial_mask_layout :
    initial.mem "M" 13 = Cell.mk .fp16 (some (5 : ℝ)) ∧
    initial.mem "M" 14 = Cell.mk .fp16 (some (6 : ℝ)) ∧
    initial.mem "M" 26 = Cell.mk .fp16 (some (26 : ℝ)) ∧
    initial.mem "M" 27 = Cell.mk .fp16 (some (28 : ℝ)) := by
  have hMX : ("M" : String) ≠ "X" := by decide
  norm_num [initial, hMX]
  all_goals repeat' constructor

end
#axiomsClean initial_mask_layout
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution

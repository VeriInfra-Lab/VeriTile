import bench.tritonbench_g.rms_rbe_matmul.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- Actual real arithmetic for the exercised backend operations. Dot is a
matrix product, reduction is a sum, power is integer power and E5M2 decoding
keeps nonfinite values absent. Identity casts do not model IEEE rounding. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => WithBot.realPow a b
  libdevicePowi := fun _ => some WithBot.realPowi
  unary := fun _ op a => match op with
    | .rsqrt => WithBot.realRsqrt a
    | .sigmoid => WithBot.realSigmoid a
    | _ => none
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => none
  float8E5Bits := some (fun _ bits => bits.decodeRat.map (fun q : ℚ => (q : ℝ)))
  fp32Load := id
  reduceMax := fun _ {_} ax keep v => ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data
  dotTyped := fun _ left right out _ {batch} {_} {_} {_} a b =>
    some (fun i => ofFloat out ((Tile.dot batch
      ⟨fun j => toFloat left (a j)⟩ ⟨fun j => toFloat right (b j)⟩).data i))

/-- Actual fp16 X with finite fp16 or integer-byte weights, a nonzero batch
id and non-unit independent strides. Unwritten cells have an integer tag. -/
def initial (fp8 : Bool) : State (WithBot ℝ) where
  mem := fun r _ => if r = "X" then .mk .fp16 (some 1)
    else if r = "W" then if fp8 then .mk .int 64 else .mk .fp16 (some 2)
    else if r = "R" then if fp8 then .mk .int 60 else .mk .fp16 (some 1)
    else .mk .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- One actual ceiling-counted iteration includes the K=1/BK=2 padding lane
and masks the N=1/BN=2 output tail. The rotary flag remains present and unused. -/
def kernelExample (fp8 variant : Bool) : ComputeKernel :=
  TypedSource.kernel .fp16 fp8 variant "X" "W" "R" "Y"
    1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 3 Bool.true 10000 2


end
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution

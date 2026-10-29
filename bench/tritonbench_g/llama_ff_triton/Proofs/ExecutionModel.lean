import bench.tritonbench_g.llama_ff_triton.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 16000000
set_option maxRecDepth 6000
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

/-- PID zero reaches a partial two-column tile with one active output lane.
Both loaded K lanes equal one, including the unmasked K=1 padding lane. -/
def initial (fp8 : Bool) : State (WithBot ℝ) where
  mem := fun r _ => if r = "A" then .mk .fp16 (some 1)
    else if r = "W1" ∨ r = "W3" ∨ r = "R" then
      if fp8 then .mk .int 60 else .mk .fp16 (some 1)
    else .mk .int 37
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Direct model call with one nonempty K iteration; native wrapper block sizes are
16/16/64. The source's nondivisible-K behavior is intentionally exercised. -/
def kernelExample (fp8 variant : Bool) : ComputeKernel :=
  TypedSource.kernel fp8 variant "A" "W1" "W3" "Y" "R"
    1 1 1 9 2 7 2 7 2 7 2 3 1 2 2 2


end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution

import bench.tritonbench_g.rotary_emb.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FullSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- Exact-real arithmetic witness backend, not an IEEE rounding model. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r ↦ some r
  negInf := none
  binary := fun _ _ op a b ↦ match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ _ _ ↦ some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ ↦ fun _ ↦ none
  reduceSum := fun _ {_} _ _ _ ↦ fun _ ↦ some 0

def payload (r : RegionName) (o : Nat) : ℝ :=
  if r="Q" then
    if o∈[220,224] then 1 else if o∈[222,226] then 2 else 7
  else if r="K" then
    if o∈[440,446,460,466] then 4 else if o∈[443,449,463,469] then 5 else 7
  else if r="Cos" then
    if o∈[100,108] then 3 else if o∈[104,112] then 7 else 17
  else if r="Sin" then
    if o∈[100,108] then 5 else if o∈[104,112] then 11 else 19
  else 7

def initial (flip : Bool) : State (WithBot ℝ) where
  mem := fun r o ↦
    if (r="Q" ∧ ¬flip) ∨ (r="K" ∧ flip) then .mk .fp16 (ofFloat .fp16 (some (payload r o)))
    else if (r="Cos" ∧ ¬flip) ∨ (r="Sin" ∧ flip) then .mk .bf16 (ofFloat .bf16 (some (payload r o)))
    else .mk .real (some (payload r o))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax=0 ∨ ax=1 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

def program (v flip : Bool) : ComputeKernel :=
  kernel (if flip then .fp64 else .fp16) (if flip then .fp16 else .fp64)
    (if flip then .fp32 else .bf16) (if flip then .bf16 else .fp32) v
    "Q" "K" "Cos" "Sin" 100 10 2 200 20 3 50 4 999 983 3 3 4 2 2 4

end
end VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution

import bench.tritonbench_g.rms_matmul_rbe.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
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

/-- Finite source data are distinct in Q/K/V. Nonzero stage numbers record
completed stores while all unused output cells retain their integer sentinel. -/
def memory (fp8 : Bool) (stage : Nat) (r : RegionName) (o : Nat) : Cell (WithBot ℝ) :=
  if r = "X" then .mk .fp16 (some 1)
  else if r = "R" then if fp8 then .mk .int 60 else .mk .fp16 (some 1)
  else if r = "QW" then if fp8 then .mk .int 64 else .mk .fp16 (some 2)
  else if r = "KW" then if fp8 then .mk .int 66 else .mk .fp16 (some 3)
  else if r = "VW" then if fp8 then .mk .int 68 else .mk .fp16 (some 4)
  else if r = "Q" ∧ o = 0 ∧ 1 ≤ stage then .mk .fp16 (some 2)
  else if r = "K" ∧ o = 0 ∧ 2 ≤ stage then
    if fp8 then .mk .f8e5 (some 3) else .mk .fp16 (some 3)
  else if r = "V" ∧ o = 0 ∧ 3 ≤ stage then
    if fp8 then .mk .f8e5 (some 4) else .mk .fp16 (some 4)
  else .mk .int 37

def initial (fp8 : Bool) : State (WithBot ℝ) where
  mem := memory fp8 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

inductive Pass where | q | k | v
  deriving DecidableEq

def before : Pass → Nat | .q => 0 | .k => 1 | .v => 2

def pass (fp8 variant : Bool) : Pass → ComputeKernel
  | .q => TypedSource.kernel .fp16 fp8 Bool.false variant "X" "QW" "R" "Q"
      1 1 1 11 7 1 1 1 4 13 5 2 1 2 2 5 Bool.true 10000 2
  | .k => TypedSource.kernel .fp16 fp8 fp8 variant "X" "KW" "R" "K"
      1 1 1 11 7 1 2 1 4 17 7 2 1 2 2 5 Bool.true 10000 2
  | .v => TypedSource.kernel .fp16 fp8 fp8 variant "X" "VW" "R" "V"
      1 1 1 11 7 1 3 1 4 19 9 2 1 2 2 5 Bool.false 10000 2

/-- Actual source QKV with a nonempty iteration and nondivisible K=1/BK=2. -/
def kernelExample (fp8 variant : Bool) : ComputeKernel :=
  TypedSource.qkv .fp16 fp8 variant "X" "QW" "KW" "VW" "R" "Q" "K" "V"
    1 1 1 11 7 1 1 1 2 1 3 1 4 13 5 2 17 7 2 19 9 2 1 2 2 5 10000 2


/-- Canonical input memory and ids, with arbitrary entry registers and other
metadata retained. The complete pass theorem separately relates this state to
its caller's state by equality, avoiding dependent casts inside numeric stages. -/
def startState (fp8 : Bool) (p : Pass) (s : State (WithBot ℝ)) : State (WithBot ℝ) :=
  { s with mem := memory fp8 (before p), pids := fun _ => 0 }

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution

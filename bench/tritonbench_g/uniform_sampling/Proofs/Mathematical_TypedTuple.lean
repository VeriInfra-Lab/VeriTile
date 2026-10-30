import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_TypedRandom
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_TupleExecution

/-! Typed execution of all five operand snapshots and all four Philox fields.
The local register conditions describe the inputs to this source statement;
they do not assume its execution or any precomputed random result.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedTuple
open VeriTile.Triton ScalarSpec RandomDomain TypedSource FP.Structural
open TupleExecution (fieldName fieldOp)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def captureBody (B : Nat) : List ComputeStmt := [
  .assign .int [] "__veritile_direct_random_0" (.alg (.ref .int [] "philox_seed")),
  .assign .int [B] "__veritile_direct_random_1" (.alg (.castNatToInt (.ref .nat [B] "c0"))),
  .assign .int [] "__veritile_direct_random_2" (.alg (.castNatToInt (.ref .nat [] "c1"))),
  .assign .int [B] "__veritile_direct_random_3" (.alg (.castNatToInt (.ref .nat [B] "_O"))),
  .assign .int [B] "__veritile_direct_random_4" (.alg (.castNatToInt (.ref .nat [B] "_O")))]

def capturedState {α : Type} {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : State α) : State α :=
  let s := s.setReg "__veritile_direct_random_0" .int [] (fun _ => seed)
  let s := s.setReg "__veritile_direct_random_1" .int [B]
    (fun i => ((counterLow offset (i4 i)).toNat : Int))
  let s := s.setReg "__veritile_direct_random_2" .int []
    (fun _ => ((counterHigh offset).toNat : Int))
  let s := s.setReg "__veritile_direct_random_3" .int [B] (fun _ => 0)
  s.setReg "__veritile_direct_random_4" .int [B] (fun _ => 0)

structure Captured {α : Type} {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : State α) : Prop where
  seedValue : s.regs .int [] "__veritile_direct_random_0" = some (fun _ => seed)
  low : s.regs .int [B] "__veritile_direct_random_1" =
    some (fun i => ((counterLow offset (i4 i)).toNat : Int))
  high : s.regs .int [] "__veritile_direct_random_2" =
    some (fun _ => ((counterHigh offset).toNat : Int))
  zero2 : s.regs .int [B] "__veritile_direct_random_3" = some (fun _ => 0)
  zero3 : s.regs .int [B] "__veritile_direct_random_4" = some (fun _ => 0)

theorem captured_values {α : Type} {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : State α) : Captured seed offset i4 (capturedState seed offset i4 s) := by
  constructor <;> simp [capturedState]

theorem captures_exec {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat) (s : State α)
    (hs : s.regs .int [] "philox_seed" = some (fun _ => seed))
    (h0 : s.regs .nat [B] "c0" = some (fun i => (counterLow offset (i4 i)).toNat))
    (h1 : s.regs .nat [] "c1" = some (fun _ => (counterHigh offset).toNat))
    (hz : s.regs .nat [B] "_O" = some (fun _ => 0)) :
    run M (captureBody B) s = some (capturedState seed offset i4 s) := by
  simp [captureBody, run, step, evalExpr, evalOp_unfold, hs, h0, h1, hz, capturedState]

def fieldStmt (B : Nat) (f : Random.Direct.Field) : ComputeStmt :=
  .assign .nat ((Random.Direct.sameShapes [B]).at f) (fieldName f) (.alg (fieldOp B f))

def fieldState {α : Type} {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (s : State α) : State α :=
  s.setReg (fieldName f) .nat ((Random.Direct.sameShapes [B]).at f) (wordField seed offset i4 f)

theorem field_eval {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (s : State α)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (h : Captured seed offset i4 s) :
    FP.Structural.evalOp M none (fieldOp B f) s = some (wordField seed offset i4 f) := by
  exact TypedRandom.eval_philox seed offset i4 f hseed M none s _ _ _ _ _
    (by simpa only [evalOp_unfold] using h.seedValue)
    (by simpa only [evalOp_unfold] using h.low)
    (by simpa only [evalOp_unfold] using h.high)
    (by simpa only [evalOp_unfold] using h.zero2)
    (by simpa only [evalOp_unfold] using h.zero3)

theorem field_step {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (s : State α)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (h : Captured seed offset i4 s) :
    step M (fieldStmt B f) s = some (fieldState seed offset i4 f s) := by
  simp only [fieldStmt, step, evalExpr]
  rw [field_eval M seed offset i4 f s hseed h]
  rfl

theorem field_preserves_captures {α : Type} {B : Nat} (seed offset : Int)
    (i4 : TileIndex [B] → Nat) (f : Random.Direct.Field) (s : State α)
    (h : Captured seed offset i4 s) :
    Captured seed offset i4 (fieldState seed offset i4 f s) := by
  cases f <;> constructor
  all_goals first
    | simpa [fieldState, fieldName] using h.seedValue
    | simpa [fieldState, fieldName] using h.low
    | simpa [fieldState, fieldName] using h.high
    | simpa [fieldState, fieldName] using h.zero2
    | simpa [fieldState, fieldName] using h.zero3

def outputBody (B : Nat) : List ComputeStmt :=
  [fieldStmt B .c0, fieldStmt B .c1, fieldStmt B .c2, fieldStmt B .c3]

def tupleBlock (B : Nat) : ComputeStmt :=
  .ifThen (.constBool Bool.true) (captureBody B ++ outputBody B)

def outputState {α : Type} {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : State α) : State α :=
  fieldState seed offset i4 .c3 (fieldState seed offset i4 .c2
    (fieldState seed offset i4 .c1 (fieldState seed offset i4 .c0 s)))

theorem outputs_exec {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : State α) (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (h : Captured seed offset i4 s) :
    run M (outputBody B) s = some (outputState seed offset i4 s) := by
  have h0 := field_preserves_captures seed offset i4 .c0 s h
  have h1 := field_preserves_captures seed offset i4 .c1 _ h0
  have h2 := field_preserves_captures seed offset i4 .c2 _ h1
  simp only [outputBody, run]
  rw [field_step M seed offset i4 .c0 s hseed h]
  dsimp only [bind, Option.bind]
  rw [field_step M seed offset i4 .c1 _ hseed h0]
  dsimp only [bind, Option.bind]
  rw [field_step M seed offset i4 .c2 _ hseed h1]
  dsimp only [bind, Option.bind]
  rw [field_step M seed offset i4 .c3 _ hseed h2]
  rfl

theorem source_slice (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    ((source storage swapped y N seed offset lower upper B hB).surfaceBody.drop 10).take 1 =
      [tupleBlock B] := by
  cases storage <;> cases swapped <;> rfl

theorem tuple_exec {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : State α) (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (hs : s.regs .int [] "philox_seed" = some (fun _ => seed))
    (h0 : s.regs .nat [B] "c0" = some (fun i => (counterLow offset (i4 i)).toNat))
    (h1 : s.regs .nat [] "c1" = some (fun _ => (counterHigh offset).toNat))
    (hz : s.regs .nat [B] "_O" = some (fun _ => 0)) :
    step M (tupleBlock B) s =
      some (outputState seed offset i4 (capturedState seed offset i4 s)) := by
  simp only [tupleBlock, step, evalOp_unfold, bind, Option.bind, if_true]
  rw [run_append, captures_exec M seed offset i4 s hs h0 h1 hz,
    Option.bind_some, outputs_exec M seed offset i4 _ hseed (captured_values seed offset i4 s)]

theorem output_register {α : Type} {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : State α) (f : Random.Direct.Field) :
    (outputState seed offset i4 s).regs .nat [B] (fieldName f) =
      some (fun i => (ScalarSpec.word seed offset (i4 i) f).toNat) := by
  cases f <;> simp [outputState, fieldState, fieldName, Random.Direct.sameShapes,
    Random.Direct.Shapes.at, wordField]
  all_goals funext i; rfl

#axiomsClean captured_values
#axiomsClean captures_exec
#axiomsClean field_eval
#axiomsClean field_step
#axiomsClean field_preserves_captures
#axiomsClean outputs_exec
#axiomsClean source_slice
#axiomsClean tuple_exec
#axiomsClean output_register

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedTuple

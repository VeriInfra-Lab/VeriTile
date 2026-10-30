import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_CounterExecution

/-! Successful execution of the complete Philox tuple block. The five input
operands are captured before any of the four outputs is written. Register
premises describe the local incoming operands, not an assumed execution.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TupleExecution
open VeriTile.Triton ScalarSpec RandomDomain
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def captureBody (B : Nat) : List Stmt := [
  .assign .int [] "__veritile_direct_random_0" (.ref .int [] "philox_seed"),
  .assign .int [B] "__veritile_direct_random_1" (.castNatToInt (.ref .nat [B] "c0")),
  .assign .int [] "__veritile_direct_random_2" (.castNatToInt (.ref .nat [] "c1")),
  .assign .int [B] "__veritile_direct_random_3" (.castNatToInt (.ref .nat [B] "_O")),
  .assign .int [B] "__veritile_direct_random_4" (.castNatToInt (.ref .nat [B] "_O"))]

def capturedState {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) : BlockState :=
  let s := s.setReg "__veritile_direct_random_0" .int [] (Tile.scalar seed)
  let s := s.setReg "__veritile_direct_random_1" .int [B]
    ⟨fun i => (counterLow offset (i4 i)).toNat⟩
  let s := s.setReg "__veritile_direct_random_2" .int []
    (Tile.scalar (counterHigh offset).toNat)
  let s := s.setReg "__veritile_direct_random_3" .int [B] ⟨fun _ => 0⟩
  s.setReg "__veritile_direct_random_4" .int [B] ⟨fun _ => 0⟩

structure Captured {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) : Prop where
  seedValue : s.regs .int [] "__veritile_direct_random_0" = some (Tile.scalar seed)
  low : s.regs .int [B] "__veritile_direct_random_1" =
    some ⟨fun i => (counterLow offset (i4 i)).toNat⟩
  high : s.regs .int [] "__veritile_direct_random_2" =
    some (Tile.scalar (counterHigh offset).toNat)
  zero2 : s.regs .int [B] "__veritile_direct_random_3" = some ⟨fun _ => 0⟩
  zero3 : s.regs .int [B] "__veritile_direct_random_4" = some ⟨fun _ => 0⟩

theorem captured_values {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) : Captured seed offset i4 (capturedState seed offset i4 s) := by
  constructor <;> simp [capturedState]

theorem captures_exec {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState)
    (hs : s.regs .int [] "philox_seed" = some (Tile.scalar seed))
    (h0 : s.regs .nat [B] "c0" = some ⟨fun i => (counterLow offset (i4 i)).toNat⟩)
    (h1 : s.regs .nat [] "c1" = some (Tile.scalar (counterHigh offset).toNat))
    (hz : s.regs .nat [B] "_O" = some ⟨fun _ => 0⟩) :
    stepStmts (captureBody B) s = some (capturedState seed offset i4 s) := by
  simp [-Tile.scalar_eta, captureBody, stepStmts, stepStmt, evalOp.eq_def, hs, h0, h1, hz,
    capturedState, Tile.scalar]

def fieldName : Random.Direct.Field → RegName
  | .c0 => "r0" | .c1 => "r1" | .c2 => "r2" | .c3 => "r3"

def fieldOp (B : Nat) (f : Random.Direct.Field) :
    Op .nat ((Random.Direct.sameShapes [B]).at f) :=
  .philox4x32 profile (Random.Direct.uniformPlan [B] 9) f
    (.ref .int [] "__veritile_direct_random_0")
    (.ref .int [B] "__veritile_direct_random_1")
    (.ref .int [] "__veritile_direct_random_2")
    (.ref .int [B] "__veritile_direct_random_3")
    (.ref .int [B] "__veritile_direct_random_4")

def fieldStmt (B : Nat) (f : Random.Direct.Field) : Stmt :=
  .assign .nat ((Random.Direct.sameShapes [B]).at f) (fieldName f) (fieldOp B f)

def fieldState {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (s : BlockState) : BlockState :=
  s.setReg (fieldName f) .nat ((Random.Direct.sameShapes [B]).at f)
    ⟨wordField seed offset i4 f⟩

theorem field_eval {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (s : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (h : Captured seed offset i4 s) :
    evalOp (fieldOp B f) s = some ⟨wordField seed offset i4 f⟩ := by
  exact eval_philox seed offset i4 f hseed s _ _ _ _ _
    (by simpa only [evalOp.eq_def] using h.seedValue)
    (by simpa only [evalOp.eq_def] using h.low)
    (by simpa only [evalOp.eq_def] using h.high)
    (by simpa only [evalOp.eq_def] using h.zero2)
    (by simpa only [evalOp.eq_def] using h.zero3)

theorem field_step {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (s : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (h : Captured seed offset i4 s) :
    stepStmt (fieldStmt B f) s = some (fieldState seed offset i4 f s) := by
  simp only [fieldStmt, stepStmt]
  rw [field_eval seed offset i4 f s hseed h]
  rfl

theorem field_preserves_captures {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (s : BlockState) (h : Captured seed offset i4 s) :
    Captured seed offset i4 (fieldState seed offset i4 f s) := by
  cases f <;> constructor
  all_goals first
    | simpa [fieldState, fieldName] using h.seedValue
    | simpa [fieldState, fieldName] using h.low
    | simpa [fieldState, fieldName] using h.high
    | simpa [fieldState, fieldName] using h.zero2
    | simpa [fieldState, fieldName] using h.zero3

def outputBody (B : Nat) : List Stmt :=
  [fieldStmt B .c0, fieldStmt B .c1, fieldStmt B .c2, fieldStmt B .c3]

def outputState {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) : BlockState :=
  fieldState seed offset i4 .c3 (fieldState seed offset i4 .c2
    (fieldState seed offset i4 .c1 (fieldState seed offset i4 .c0 s)))

theorem outputs_exec {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (h : Captured seed offset i4 s) :
    stepStmts (outputBody B) s = some (outputState seed offset i4 s) := by
  have h0 := field_preserves_captures seed offset i4 .c0 s h
  have h1 := field_preserves_captures seed offset i4 .c1 _ h0
  have h2 := field_preserves_captures seed offset i4 .c2 _ h1
  simp only [outputBody, stepStmts]
  rw [field_step seed offset i4 .c0 s hseed h]
  dsimp only
  rw [field_step seed offset i4 .c1 _ hseed h0]
  dsimp only
  rw [field_step seed offset i4 .c2 _ hseed h1]
  dsimp only
  rw [field_step seed offset i4 .c3 _ hseed h2]
  rfl

theorem tupleBlock_eq (B : Nat) : Algorithm.tupleBlock B =
    .ifThen (.constBool Bool.true) (captureBody B ++ outputBody B) := rfl

theorem tuple_exec {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (hs : s.regs .int [] "philox_seed" = some (Tile.scalar seed))
    (h0 : s.regs .nat [B] "c0" = some ⟨fun i => (counterLow offset (i4 i)).toNat⟩)
    (h1 : s.regs .nat [] "c1" = some (Tile.scalar (counterHigh offset).toNat))
    (hz : s.regs .nat [B] "_O" = some ⟨fun _ => 0⟩) :
    stepStmt (Algorithm.tupleBlock B) s =
      some (outputState seed offset i4 (capturedState seed offset i4 s)) := by
  rw [tupleBlock_eq]
  simp only [stepStmt, evalOp.eq_def, Tile.scalar, bind, Option.bind, if_true]
  rw [CounterExecution.steps_append, captures_exec seed offset i4 s hs h0 h1 hz,
    Option.bind_some, outputs_exec seed offset i4 _ hseed (captured_values seed offset i4 s)]

theorem outputState_frame {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) :
    (outputState seed offset i4 (capturedState seed offset i4 s)).mem = s.mem ∧
    (outputState seed offset i4 (capturedState seed offset i4 s)).pids = s.pids ∧
    (outputState seed offset i4 (capturedState seed offset i4 s)).undef = s.undef ∧
    (outputState seed offset i4 (capturedState seed offset i4 s)).numPids = s.numPids :=
  ⟨rfl, rfl, rfl, rfl⟩

#axiomsClean captured_values
#axiomsClean captures_exec
#axiomsClean field_eval
#axiomsClean field_step
#axiomsClean field_preserves_captures
#axiomsClean outputs_exec
#axiomsClean tupleBlock_eq
#axiomsClean tuple_exec
#axiomsClean outputState_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TupleExecution

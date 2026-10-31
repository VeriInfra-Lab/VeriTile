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

def outputState {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) : BlockState :=
  fieldState seed offset i4 .c3 (fieldState seed offset i4 .c2
    (fieldState seed offset i4 .c1 (fieldState seed offset i4 .c0 s)))

theorem outputState_frame {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (s : BlockState) :
    (outputState seed offset i4 (capturedState seed offset i4 s)).mem = s.mem ∧
    (outputState seed offset i4 (capturedState seed offset i4 s)).pids = s.pids ∧
    (outputState seed offset i4 (capturedState seed offset i4 s)).undef = s.undef ∧
    (outputState seed offset i4 (capturedState seed offset i4 s)).numPids = s.numPids :=
  ⟨rfl, rfl, rfl, rfl⟩

#axiomsClean captured_values
#axiomsClean field_eval
#axiomsClean field_preserves_captures
#axiomsClean outputState_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TupleExecution

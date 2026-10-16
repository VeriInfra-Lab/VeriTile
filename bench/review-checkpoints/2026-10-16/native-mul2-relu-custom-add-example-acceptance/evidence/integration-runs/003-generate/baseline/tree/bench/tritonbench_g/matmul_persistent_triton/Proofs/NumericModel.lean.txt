import bench.tritonbench_g.matmul_persistent_triton.Proofs.FP
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- A value-sensitive exact-real backend: fused dot computes the matrix
product plus its supplied accumulator. Identity casts are a witness model,
not a statement about native IEEE rounding. -/
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
  unary := fun _ op a => match op with
    | .sqrt => WithBot.realSqrt a
    | _ => none
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => none
  fp32Load := id
  reduceMax := fun _ {_} ax keep v =>
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data
  dotAcc := fun _ left right out _ {batch} {_} {_} {_} a b c =>
    some (fun i => ofFloat out (WithBot.realAdd
      ((Tile.dot batch ⟨fun j => toFloat left (a j)⟩
        ⟨fun j => toFloat right (b j)⟩).data i)
      (toFloat out (c i))))

def channel : TypedSource.Storage → FloatDType
  | .fp16 => .fp16
  | .f8e4 => .f8e4

/-- PID 1 handles tile IDs 1 and 3. K=3/BK=2 exercises a masked K tail;
M=N=3/BM=BN=2 reaches both row and column boundaries and a reset between
persistent output tiles. Independent strides preserve visible gaps. -/
def sample : Parameters where
  A := "A"; B := "B"; C := "C"
  M := 3; N := 3; K := 3
  sam := 9; sak := 2; sbk := 11; sbn := 2; scm := 7; scn := 2
  BM := 2; BN := 2; BK := 2; GM := 2; SMS := 2

def initial (storage : TypedSource.Storage) : State (WithBot ℝ) where
  mem := fun r o => if r = "A" then
      .mk (channel storage).toTileDType (ofFloat (channel storage) (some ((o / 9 + 1 : Nat) : ℝ)))
    else if r = "B" then
      .mk (channel storage).toTileDType
        (ofFloat (channel storage) (some ((o / 11 + (o % 11) / 2 + 1 : Nat) : ℝ)))
    else .mk .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution

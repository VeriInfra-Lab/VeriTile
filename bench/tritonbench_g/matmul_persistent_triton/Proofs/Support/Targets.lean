import bench.tritonbench_g.matmul_persistent_triton.Proofs.Real
import bench.tritonbench_g.matmul_persistent_triton.Proofs.DotPlans

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Targets
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 7000000

/-- Only caller storage and matrix coordinates define the target. -/
def left {α : Type} (zero : α) (p : Problem) (b : Buffers) (entry : Memory α)
    (rowBlock iteration : Nat) (row : Fin 128) (lane : Fin p.profile.blockK) : α :=
  let k := iteration*p.profile.blockK+lane.val
  if k<p.K then entry (b.A.address (clamp p.M (rowBlock*128+row.val)) k) else zero

def right {α : Type} (zero : α) (p : Problem) (b : Buffers) (entry : Memory α)
    (columnBlock iteration : Nat) (lane : Fin p.profile.blockK) (column : Fin 256) : α :=
  let k := iteration*p.profile.blockK+lane.val
  if k<p.K then entry (b.B.address k (clamp p.N (columnBlock*256+column.val))) else zero

def accReference {α : Type} (P : Source.Arithmetic α α) (p : Problem) (b : Buffers)
    (entry : Memory α) (rowBlock columnBlock : Nat) : Nat → Source.Acc α
  | 0 => fun _ _ => P.accumulatorZero
  | count+1 => P.dot p.profile.blockK (left P.inputZero p b entry rowBlock count)
      (right P.inputZero p b entry columnBlock count) (accReference P p b entry rowBlock columnBlock count)

def result {α : Type} (P : Source.Arithmetic α α) (p : Problem) (b : Buffers)
    (entry : Memory α) (row column : Nat) : α :=
  P.cast p.profile (accReference P p b entry (row/128) (column/256) p.kTiles
    ⟨row%128,Nat.mod_lt _ (by decide)⟩ ⟨column%256,Nat.mod_lt _ (by decide)⟩)

theorem accumulation {α : Type} (P : Source.Arithmetic α α) (p : Problem) (b : Buffers)
    (entry : Memory α) (tile count : Nat) :
    Source.accumulator P p b entry tile count=
      accReference P p b entry (p.tileRow tile) (p.tileColumn tile) count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    simp only [Source.accumulator,accReference,ih]
    congr 1
    · funext row k
      simp only [Source.loadA,left,Problem.row,Problem.k,clamp]
      split_ifs <;> rfl
    · funext k column
      simp only [Source.loadB,right,Problem.column,Problem.k,clamp]
      split_ifs <;> rfl

theorem source_result {α : Type} (P : Source.Arithmetic α α) (p : Problem) (b : Buffers)
    (entry : Memory α) (tile : Nat) (lane : Source.Lane) :
    P.cast p.profile (Source.accumulator P p b entry tile p.kTiles lane.1 lane.2)=
      result P p b entry (p.row tile lane.1.val) (p.column tile lane.2.val) := by
  rw [accumulation]
  simp only [result,Source.row_recover,Source.column_recover]
  congr 2 <;> apply Fin.ext
  · simp [Problem.row,Nat.add_mod,Nat.mod_eq_of_lt lane.1.isLt]
  · simp [Problem.column,Nat.add_mod,Nat.mod_eq_of_lt lane.2.isLt]

theorem real (p : Problem) (b : Buffers) (entry : Memory ℝ) (tile : Nat)
    (lane : Source.Lane) (active : Source.active p tile lane) :
    result Real.arithmetic p b entry (p.row tile lane.1.val) (p.column tile lane.2.val)=
      RealMath.formula p.K (Real.inputs b entry) (p.row tile lane.1.val) (p.column tile lane.2.val) := by
  rw [← source_result]
  exact Real.completed p b entry tile lane active

#axiomsClean accumulation
#axiomsClean source_result
#axiomsClean real
#specNonCircular result avoiding [Source.run,Source.accumulator]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Targets

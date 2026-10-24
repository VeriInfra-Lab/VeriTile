import bench.tritonbench_g.fused_recurrent_retention.Proofs.Model
import VeriTile.Meta.StatementAudit

/-! Small actual-evaluator lemmas for the forward producer. Register premises
describe one reached internal stage; complete execution must establish them
from the source initialization and loop invariant. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Primitives
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def vector (n : Nat) (f : Fin n → ℝ) : Tile .real [n] :=
  Tile.vec fun j => some (f j)
def matrix (m n : Nat) (f : Fin m → Fin n → ℝ) : Tile .real [m,n] :=
  Tile.mat fun j k => some (f j k)
def pointers (n : Nat) (r : RegionName) (addr : Fin n → Nat) : Tile .ptr [n] :=
  Tile.vec fun j => ⟨r, addr j⟩
def maskTile (n : Nat) (active : Fin n → Prop) [DecidablePred active] : Tile .bool [n] :=
  Tile.vec fun j => decide (active j)

def loadOp (n : Nat) (ptr mask : RegName) : Op .real [n] :=
  .load .real (.ptr (.ref .ptr [n] ptr))
    (.maskOther (.ref .bool [n] mask) (.broadcast (.const 0) [n]))

theorem masked_load (n : Nat) (r : RegionName) (ptr mask : RegName) (addr : Fin n → Nat)
    (active : Fin n → Prop) [DecidablePred active] (u : BlockState)
    (hp : u.regs .ptr [n] ptr = some (pointers n r addr))
    (hm : u.regs .bool [n] mask = some (maskTile n active)) :
    evalOp (loadOp n ptr mask) u =
      some (vector n (fun j => if active j then u.readMem r (addr j) else 0)) := by
  simp only [loadOp, evalOp.eq_def, hp, hm, Option.bind, bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  by_cases hj : active j <;>
    simp [pointers, maskTile, vector, Tile.vec, BlockState.readMemValue_real, hj]

theorem scale_eval (n : Nat) (x : Op .real [n]) (u : BlockState)
    (f : Fin n → ℝ) (c : ℝ) (hx : evalOp x u = some (vector n f)) :
    evalOp (.mul .real .scalarR x (.const c)) u = some (vector n (fun j => f j*c)) := by
  rw [evalOp_mul, hx, evalOp_const]
  simp only [Option.bind, bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [vector, tile_elementwise, Broadcast.leftIndex, Broadcast.rightIndex]

def stateOp (m n : Nat) : Op .real [m,n] :=
  .add .real (.consSame (.consSame .nil))
    (.mul .real .scalarL (.ref .real [] "b_b") (.ref .real [m,n] "h"))
    (.mul .real (.consL (.consR .nil))
      (.expandDim ⟨0,by simp⟩ (.ref .real [n] "_k"))
      (.expandDim ⟨1,by simp⟩ (.ref .real [m] "_v")))

theorem state_update (m n : Nat) (u : BlockState) (b : ℝ)
    (h : Fin m → Fin n → ℝ) (k : Fin n → ℝ) (v : Fin m → ℝ)
    (hb : u.regs .real [] "b_b" = some (Tile.scalar (some b)))
    (hh : u.regs .real [m,n] "h" = some (matrix m n h))
    (hk : u.regs .real [n] "_k" = some (vector n k))
    (hv : u.regs .real [m] "_v" = some (vector m v)) :
    evalOp (stateOp m n) u =
      some (matrix m n (fun j e => b*h j e + k e*v j)) := by
  simp only [stateOp, evalOp.eq_def, hb, hh, hk, hv, Option.bind, bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,e,⟨⟩⟩
  simp [matrix, vector, tile_elementwise, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

def productOp (m n : Nat) : Op .real [m,n] :=
  .mul .real (.consR (.consSame .nil)) (.ref .real [m,n] "h")
    (.expandDim ⟨0,by simp⟩ (.ref .real [n] "_q"))

theorem output_product (m n : Nat) (u : BlockState)
    (h : Fin m → Fin n → ℝ) (q : Fin n → ℝ)
    (hh : u.regs .real [m,n] "h" = some (matrix m n h))
    (hq : u.regs .real [n] "_q" = some (vector n q)) :
    evalOp (productOp m n) u = some (matrix m n (fun j e => h j e*q e)) := by
  simp only [productOp, evalOp.eq_def, hh, hq, Option.bind, bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,e,⟨⟩⟩
  simp [matrix, vector, tile_elementwise, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

theorem sum_rows (m n : Nat) (u : BlockState) (f : Fin m → Fin n → ℝ)
    (hf : u.regs .real [m,n] "_o" = some (matrix m n f)) :
    evalOp (.reduceSum ⟨1,by simp⟩ Bool.false (.ref .real [m,n] "_o")) u =
      some (vector m (fun j => ∑ k : Fin n, f j k)) := by
  simp only [evalOp.eq_def, hf, Option.bind, bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp only [Tile.reduceSum_false, Tile.reduceSumDrop_data, matrix, vector,
    Tile.mat, Tile.vec, TileShape.axisDim, TileShape.insertAxisIndex]
  change (∑ k : Fin n, ((f j k : ℝ) : WithBot ℝ)) = some (∑ k : Fin n, f j k)
  rw [← WithBot.coe_sum]
  rfl

theorem advance_pointer (n : Nat) (r : RegionName) (ptr : RegName) (addr : Fin n → Nat)
    (step : Nat) (u : BlockState)
    (hp : u.regs .ptr [n] ptr = some (pointers n r addr)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [n] ptr) (.constNat step)) u =
      some (pointers n r (fun j => addr j + step)) := by
  simp only [evalOp.eq_def, hp, Option.bind, bind, Option.pure_def]
  rfl

#axiomsClean masked_load
#axiomsClean scale_eval
#axiomsClean state_update
#axiomsClean output_product
#axiomsClean sum_rows
#axiomsClean advance_pointer
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Primitives

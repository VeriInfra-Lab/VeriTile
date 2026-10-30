import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealProgram
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealMemory
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.SignedIndex

/-! Successful evaluation of every statement in one complete descending
iteration. The carry and unspecified input oracle are arbitrary. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealStep
open VeriTile.Triton RealValues RealProgram
noncomputable section
set_option maxHeartbeats 2400000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def beforeStore (T BT row : Nat) (s o : RegionName) (c : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) : BlockState :=
  let u0 := u.setReg "i_t" .int [] (Tile.scalar (c : Int))
  let u1 := u0.setReg "p_s" .blockPtr [BT] (pointer T BT row s c)
  let u2 := u1.setReg "p_o" .blockPtr [BT] (pointer T BT row o c)
  let u3 := u2.setReg "b_s" .real [BT] (inputTile BT c f)
  let u4 := u3.setReg "b_zz" .real [] (Tile.scalar (some (chunkSum BT c f)))
  let u5 := u4.setReg "b_z" .real [] (Tile.scalar (some (acc + chunkSum BT c f)))
  u5.setReg "b_o" .real [BT] (outputTile BT c f (acc + chunkSum BT c f))

def result (T BT row : Nat) (s o : RegionName) (c : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) : BlockState :=
  RealMemory.scatter T BT row o c f (acc + chunkSum BT c f)
    (beforeStore T BT row s o c f acc u)

theorem index_eval (j : Int) (u : BlockState) (hj : j < 1)
    (hr : u.regs .int [] "__rev_i_t" = some (Tile.scalar j)) :
    evalOp (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")) u =
      some (Tile.scalar ((-j).toNat : Int)) := by
  rw [evalOp_sub, constInt_eval, evalOp_ref, hr]
  rw [SignedIndex.int_index_of_nat j hj]
  apply congrArg some
  apply Tile.ext
  rintro ⟨⟩
  change (0 : Int) - j = -j
  omega

theorem pointer_eval (swapped : Bool) (T BT row : Nat) (r : RegionName)
    (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar row))
    (ht : u.regs .int [] "i_t" = some (Tile.scalar (c : Int))) :
    evalOp (pointerOp swapped T BT r) u = some (pointer T BT row r c) := by
  have hmul : ((c : Int) * (BT : Int)).toNat = c * BT := by
    rw [← Int.natCast_mul]
    rfl
  cases swapped <;>
    simp [-Tile.scalar_eta, pointerOp, TypedProgram.offsetOp, evalOp, hh, ht,
      Option.bind, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul, pointer, hmul, Nat.mul_comm]

theorem executes (swapped : Bool) (T BT row : Nat) (s o : RegionName)
    (input padding : Nat → ℝ) (acc : ℝ) (j : Int) (u : BlockState) (hj : j < 1)
    (hr : u.regs .int [] "__rev_i_t" = some (Tile.scalar j))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar row))
    (ha : u.regs .real [] "b_z" = some (Tile.scalar (some acc)))
    (hm : ∀ off, u.readMem s off = input off)
    (hu : ∀ off, u.undef s off = padding off) :
    stepStmts (loopBody swapped T BT s o) u =
      some (result T BT row s o (-j).toNat (value T row input padding) acc u) := by
  unfold loopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (index_eval j u hj hr))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval swapped T BT row s
    (-j).toNat _ (by simp [hh]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval Bool.false T BT row o
    (-j).toNat _ (by simp [hh]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_eval T BT row s (-j).toNat
    input padding _ (by simp) (by intro off; simp [hm]) (by intro off; simp [hu])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := []) (sum_eval BT (-j).toNat
    (value T row input padding) _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (carry_eval BT (-j).toNat
    (value T row input padding) acc _ (by simp [ha]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (output_eval BT (-j).toNat
    (value T row input padding) (acc + chunkSum BT (-j).toNat (value T row input padding)) _
    (by simp) (by simp)))]
  rw [stepStmts.cons_some (RealMemory.store_eval T BT row o (-j).toNat
    (value T row input padding) (acc + chunkSum BT (-j).toNat (value T row input padding)) _
    (by simp) (by simp))]
  rw [stepStmts.nil]
  rfl

#axiomsClean index_eval
#axiomsClean pointer_eval
#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealStep

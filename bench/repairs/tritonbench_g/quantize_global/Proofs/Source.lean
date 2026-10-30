import bench.repairs.tritonbench_g.quantize_global.Proofs.Model
namespace QuantizeGlobalRepair
open VeriTile.Triton FP.Structural QuantizeIntegerStage
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def offset (s : State α) (base B : Nat) (i : TileIndex [B]) := s.pids 0 * B + i.1.val + base
def word [Inhabited α] (a : Scalar α) (s : State α)
    (X Maximum : RegionName) (base B : Nat) (i : TileIndex [B]) : BitVec 8 :=
  scalar a ((s.mem X (offset s base B i)).read .real) ((s.mem Maximum 0).read .real)
def written [Inhabited α] (a : Scalar α) (s : State α)
    (X Maximum Out : RegionName) (N base B : Nat) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if offset s base B i < N then
      t.write Out (offset s base B i) (.mk .int (word a s X Maximum base B i).toInt)
    else t) s

/-- The actual full raw program invokes the dedicated precise primitive at
its original assignment site, independently of the callback implementation. -/
theorem precise_division_statement (X Maximum Out : RegionName) (N base B : Nat) :
    (source X Maximum Out N base B).surfaceBody[7]? = some
      (.assign .real [B] "normalized" (.compute (.alg .fp32
        (.divRN .scalarR (.ref .real [B] "x") (.ref .real [] "denominator"))))) := rfl

private def trunc8 (x : Int) : Int :=
  if 2 * ((x % 256).toNat % 256) < 256 then
    (((x % 256).toNat % 256 : Nat) : Int)
  else (((x % 256).toNat % 256 : Nat) : Int) - 256

private theorem trunc8_eq (x : Int) : trunc8 x=Int.bmod x 256 := by
  have hn : 0 ≤ x%256 := Int.emod_nonneg _ (by decide)
  have hl : x%256<256 := Int.emod_lt_of_pos _ (by decide)
  have ht : (x%256).toNat<256 := by omega
  have hc : (((x%256).toNat : Nat) : Int)=x%256 := Int.toNat_of_nonneg hn
  simp only [trunc8,Nat.mod_eq_of_lt ht,hc,Int.bmod]
  split_ifs <;> omega

private theorem byte_idempotent (x : Int) :
    Int.bmod (if 2 * ((x % 256).toNat % 256) < 256 then
      (((x % 256).toNat % 256 : Nat) : Int)
    else (((x % 256).toNat % 256 : Nat) : Int) - 256) 256 = Int.bmod x 256 := by
  change Int.bmod (trunc8 x) 256=Int.bmod x 256
  rw [trunc8_eq,Int.bmod_bmod]

/-- The checked AST executes every assignment, comparison, safe masked load,
div, multiplication, clamp, signed64 llrint and modular int8 store. -/
theorem source_run [Inhabited α] (rest : Algebra α) (a : Scalar α) (s : State α)
    (X Maximum Out : RegionName) (N base B : Nat) :
    ∃ t, QuantizeIntegerStage.exec (backend rest a) (some a.round64)
      (kernel X Maximum Out N base B) s = some t ∧
      t.mem = (written a s X Maximum Out N base B).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [kernel, source, ComputeKernel.surfaceBody,
    QuantizeIntegerStage.exec, QuantizeIntegerStage.run, QuantizeIntegerStage.step,
    QuantizeIntegerStage.evalExpr, QuantizeIntegerStage.store, storedInteger, byteOutput, outputRegister,
    FP.Structural.step, stepAlg, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    FP.Structural.store, backend, State.setReg, TileShape.axisDim, written, word, scalar,
    bounded, scaled, denominator, offset, ofFloat, toFloat, Region.cast,
    ComputeDType.eraseDType, bop, numeric, numericLt, numericEq, natLt, Option.bind]
  simp only [fold_memory]
  simp only [Int.ofNat_eq_natCast, ←Int.natCast_mul, ←Int.natCast_add,
    Int.toNat_natCast, Int.ofNat_lt]
  simp +contextual [show (127.0 : ℝ) = 127 by norm_num]
  congr 1
  funext mem i r o
  by_cases hi : s.pids 0 * B + i.1.val + base < N <;>
    simp only [hi,if_true,if_false,Int.toNat_natCast]
  all_goals simp only [integralMod,byte_idempotent]
  all_goals simp [hi,show (0.0 : ℝ)=0 by norm_num,show (1.0 : ℝ)=1 by norm_num]
  all_goals simp only [Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,
    Int.toNat_natCast]

#axiomsClean precise_division_statement
#axiomsClean source_run
end
end QuantizeGlobalRepair

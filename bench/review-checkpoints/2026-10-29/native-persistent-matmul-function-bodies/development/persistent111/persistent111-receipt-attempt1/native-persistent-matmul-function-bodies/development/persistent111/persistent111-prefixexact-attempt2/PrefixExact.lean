import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Registers
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.PrefixExecution
import VeriTile.Triton.Semantics.Integer

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

private def signedQuotient (x y : Int) := if x < 0 then -((-x)/y) else x/y
private def signedRemainder (x y : Int) :=
  x - if x < 0 then -(y*((-x)/y)) else y*(x/y)
private def computedCeil (n d : Nat) : Int :=
  signedQuotient n d + if signedRemainder n d = 0 then 0 else 1
private def computedGroup (n : Int) := if (8 : Int) < n then 8 else n

private theorem quotient_nat (n d : Nat) : signedQuotient n d = ((n/d : Nat) : Int) := by
  simp [signedQuotient]
private theorem remainder_nat (n d : Nat) : signedRemainder n d = ((n%d : Nat) : Int) := by
  simp only [signedRemainder,show ¬(n : Int)<0 by omega,if_false,← Int.emod_def]
  norm_cast
private theorem computedCeil_nat (n d : Nat) : computedCeil n d = (ceil n d : Int) := by
  rw [computedCeil,quotient_nat,remainder_nat]
  by_cases h : n%d=0 <;> simp [ceil,h]
  simpa only [Int.natCast_dvd_natCast,Nat.dvd_iff_mod_eq_zero] using h
private theorem computedGroup_nat (n : Nat) : computedGroup n = ((min 8 n : Nat) : Int) := by
  by_cases h : 8 < n
  · simp [computedGroup,h,min_eq_left (by omega : 8 ≤ n)]
  · simp [computedGroup,h,min_eq_right (by omega : n ≤ 8)]

theorem ceil_exact (n d : Nat) (width : d=64 ∨ d=128 ∨ d=256) :
    ceil n d=RealMath.blocks n d := by
  rw [← source_ceil n d width]
  unfold ceil
  by_cases h : n%d=0 <;> simp [h]

variable {α : Type} [Inhabited α]
private theorem rows_actual (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    (preludeExecution P p s).val.regs .int [] "num_pid_m" =
      some (fun _ => computedCeil p.problem.M 128) := by rfl
private theorem columns_actual (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    (preludeExecution P p s).val.regs .int [] "num_pid_n" =
      some (fun _ => computedCeil p.problem.N 256) := by rfl
private theorem count_actual (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    (preludeExecution P p s).val.regs .int [] "k_tiles" =
      some (fun _ => computedCeil p.problem.K p.problem.profile.blockK) := by rfl
private theorem group_actual (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    (preludeExecution P p s).val.regs .int [] "active_group_m" =
      some (fun _ => computedGroup (computedCeil p.problem.M 128)) := by rfl
private theorem span_actual (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    (preludeExecution P p s).val.regs .int [] "num_pid_in_group" =
      some (fun _ => computedGroup (computedCeil p.problem.M 128) * computedCeil p.problem.N 256) := by rfl
private theorem work_actual (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    (preludeExecution P p s).val.regs .int [] "tiles_per_SM" =
      some (fun _ => signedQuotient (computedCeil p.problem.M 128 * computedCeil p.problem.N 256) p.problem.SMS +
        if (s.pids 0 : Int) < signedRemainder (computedCeil p.problem.M 128 * computedCeil p.problem.N 256) p.problem.SMS then 1 else 0) := by rfl

/-- The source computes these signed controls; their natural schedule follows
from exact integer identities, including zero dimensions and remainder tails. -/
theorem prelude_invariant (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    Registers p (s.pids 0) (preludeExecution P p s).val ∧
    (preludeExecution P p s).val.mem=s.mem := by
  have hm : computedCeil p.problem.M 128 = (p.problem.rows : Int) := by
    rw [computedCeil_nat,ceil_exact _ _ (by omega)]; rfl
  have hn : computedCeil p.problem.N 256 = (p.problem.columns : Int) := by
    rw [computedCeil_nat,ceil_exact _ _ (by omega)]; rfl
  have hk : computedCeil p.problem.K p.problem.profile.blockK = (p.problem.kTiles : Int) := by
    rw [computedCeil_nat,ceil_exact _ _ (by cases p.problem.profile <;> simp [Storage.blockK])]; rfl
  constructor
  · constructor
    · rfl
    · rw [rows_actual,hm]
    · rw [columns_actual,hn]
    · rw [count_actual,hk]
    · rw [work_actual,hm,hn,← Int.natCast_mul,quotient_nat,remainder_nat]
      simp [Problem.work,Schedule.tilesFor,Problem.tiles]
      funext i
      norm_cast
    · rw [group_actual,hm,computedGroup_nat]
      rfl
    · rw [span_actual,hm,hn,computedGroup_nat]
      simp [Problem.group]
    · rfl
    · rfl
    · simp only [preludeExecution,State.setReg_same]
    all_goals rfl
  · rfl

#axiomsClean prelude_invariant
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core

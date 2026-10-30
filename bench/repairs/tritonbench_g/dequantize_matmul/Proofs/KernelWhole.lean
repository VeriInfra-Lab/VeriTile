import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelModels
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelEquivalence

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelWhole
open VeriTile.Triton FP.Structural KernelSource KernelExecution KernelResult KernelModels
noncomputable section

theorem same_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (order : List Program) (s : State α) :
    KernelGrid.run M p Bool.false order s = KernelGrid.run M p Bool.true order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    change (exec M p Bool.false (KernelGrid.atProgram s pid)).bind
        (KernelGrid.run M p Bool.false tail) =
      (exec M p Bool.true (KernelGrid.atProgram s pid)).bind
        (KernelGrid.run M p Bool.true tail)
    rw [KernelEquivalence.same_run]
    exact congrArg
      (fun (continuation : State α → Option (State α)) =>
        (exec M p Bool.true (KernelGrid.atProgram s pid)).bind continuation)
      (funext fun next => ih next)

theorem real_correct (rest : Algebra ℝ) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p)
    (q : Nat → BitVec 8) (scale : Nat → ℝ) (s : State ℝ) (reads : Reads p q scale s) :
    ∃ final, KernelGrid.run (realAlgebra rest) p Bool.false (KernelGrid.order p) s = some final ∧
      KernelGrid.run (realAlgebra rest) p Bool.true (KernelGrid.order p) s = some final ∧
      Result p (fun i => mathematical (q i) (scale (i % p.N))) s final := by
  obtain ⟨final, original, result⟩ := whole_correct (realAlgebra rest) p hK hN sep q scale s reads
  refine ⟨final, original, (same_run _ _ _ _).symm.trans original, ?_⟩
  simpa only [real_formula] using result

theorem native_correct (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p)
    (q : Nat → BitVec 8) (scale : Nat → Word) (s : State Word) (reads : Reads p q scale s) :
    ∃ final, KernelGrid.run (nativeAlgebra multiply rest) p Bool.false (KernelGrid.order p) s = some final ∧
      KernelGrid.run (nativeAlgebra multiply rest) p Bool.true (KernelGrid.order p) s = some final ∧
      Result p (fun i => nativeWord multiply (q i) (scale (i % p.N))) s final := by
  obtain ⟨final, original, result⟩ := whole_correct (nativeAlgebra multiply rest) p hK hN sep q scale s reads
  refine ⟨final, original, (same_run _ _ _ _).symm.trans original, ?_⟩
  simpa only [native_formula] using result

#axiomsClean same_run
#axiomsClean real_correct
#axiomsClean native_correct
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelWhole

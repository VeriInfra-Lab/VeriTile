import bench.tritonbench_g.matmul_persistent_triton.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.ZeroK
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

/-- Successful evaluation of the original sixteen-statement K=0 prefix.
The complete no-store source theorem is proved below using its actual state. -/
def prefix_run (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (hBK : 0 < p.BK) {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    {t : State α // run (engine M)
      ((({p with K := 0} : Parameters).source storage v).surfaceBody.take 16) s = some t} := by
  have hz : (0 + p.BK - 1) / p.BK = 0 := Nat.div_eq_of_lt (by omega)
  by_cases hpid : s.pids 0 < (((p.M + p.BM - 1) / p.BM) *
    ((p.N + p.BN - 1) / p.BN)) % p.SMS
  all_goals cases storage <;> cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 300000) [Parameters.source, TypedSource.kernel,
        ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
        run, step, FP.Structural.range, FP.Structural.evalExpr, evalComputeOp,
        evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
        resolvePrecision, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, FP.Structural.bop, natLt,
        ofFloat, toFloat, TileShape.axisDim, Region.cast, hpid, hz]
    try rfl

theorem prefix_values (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (hBK : 0 < p.BK) {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    let t := (prefix_run p storage v hBK M s).val
    t.mem = s.mem ∧
    t.regs .nat [] "k_tiles" = some (fun _ => 0) ∧
    ∃ n, t.regs .nat [] "tiles_per_SM" = some (fun _ => n) := by
  have hz : (p.BK - 1) / p.BK = 0 := Nat.div_eq_of_lt (by omega)
  by_cases hpid : s.pids 0 < (((p.M + p.BM - 1) / p.BM) *
    ((p.N + p.BN - 1) / p.BN)) % p.SMS
  all_goals cases storage <;> cases v
  all_goals simp [prefix_run, hpid, FP.Structural.bop, numeric, integralFloorDiv, integralMod,
    engine, Algebra.withDefaultPrecision, hz]
  all_goals
    constructor
    · funext i
      simp [FP.Structural.bop, hz]
    · refine ⟨(((p.M + p.BM - 1) / p.BM) * ((p.N + p.BN - 1) / p.BN)) / p.SMS +
        (if s.pids 0 < (((p.M + p.BM - 1) / p.BM) *
          ((p.N + p.BN - 1) / p.BN)) % p.SMS then 1 else 0), ?_⟩
      funext i
      simp [FP.Structural.bop, hpid]

theorem tail_shape (p : Parameters) (storage : TypedSource.Storage) (v : Bool) :
    (p.source storage v).surfaceBody.drop 16 =
      [.forRangeDyn "_i" (.constNat 0)
        (.mul .nat .nil (.ref .nat [] "k_tiles") (.ref .nat [] "tiles_per_SM"))
        (.constNat 1) (FPProof.loop p storage v)] := by
  cases storage <;> cases v <;> rfl

private theorem empty_range {α : Type} [Inhabited α] (M : Algebra α)
    (body : List ComputeStmt) (s : State α) :
    FP.Structural.range M "_i" 0 0 1 body s = some s := by
  rw [FP.Structural.range]
  simp only [Nat.one_ne_zero, Nat.lt_irrefl, if_false]

/-- The complete original K=0 branch preserves the complete typed memory,
including every preexisting output cell. This is an actual source execution. -/
theorem run_unchanged (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (hBK : 0 < p.BK) {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, run (engine M)
      (({p with K := 0} : Parameters).source storage v).surfaceBody s = some t ∧
      t.mem = s.mem := by
  refine ⟨(prefix_run p storage v hBK M s).val, ?_,
    (prefix_values p storage v hBK M s).1⟩
  obtain ⟨n, hn⟩ := (prefix_values p storage v hBK M s).2.2
  have hk := (prefix_values p storage v hBK M s).2.1
  conv_lhs => arg 2; rw [← List.take_append_drop 16
    (({p with K := 0} : Parameters).source storage v).surfaceBody]
  simp only [run_append, (prefix_run p storage v hBK M s).property, Option.bind_some, tail_shape]
  simp [run, step, evalOp_unfold, empty_range, hk, hn, FP.Structural.bop, numeric,
    Option.bind, engine, Algebra.withDefaultPrecision]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.ZeroK

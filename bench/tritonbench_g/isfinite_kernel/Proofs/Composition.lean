import bench.tritonbench_g.isfinite_kernel.Proofs.Header

/-! Generic source-control composition. Range induction carries an arbitrary
register invariant and an exact memory transformer. Its induction step uses
an actual successful body run; neither a successful whole run nor prepared
initial registers are assumptions of the eventual source theorem. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural
noncomputable section

def iterateMemory {α : Type}
    (update : Nat → (RegionName → Nat → Cell α) → RegionName → Nat → Cell α) :
    Nat → Nat → (RegionName → Nat → Cell α) → RegionName → Nat → Cell α
  | 0, _, mem => mem
  | n + 1, cur, mem => iterateMemory update n (cur + 1) (update cur mem)

theorem range_memory {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (ready : State α → Prop)
    (update : Nat → (RegionName → Nat → Cell α) → RegionName → Nat → Cell α)
    (hbody : ∀ j s, ready s → ∃ t,
      FP.Structural.run M body (s.setReg idx .nat [] (fun _ => j)) = some t ∧
      ready t ∧ t.mem = update j s.mem)
    (n cur : Nat) (s : State α) (hs : ready s) :
    ∃ t, FP.Structural.range M idx cur (cur + n) 1 body s = some t ∧
      ready t ∧ t.mem = iterateMemory update n cur s.mem := by
  induction n generalizing cur s with
  | zero =>
      refine ⟨s, ?_, hs, rfl⟩
      rw [FP.Structural.range]
      simp
  | succ n ih =>
      obtain ⟨t, he, ht, hm⟩ := hbody cur s hs
      obtain ⟨u, hu, hready, hmem⟩ := ih (cur + 1) t ht
      refine ⟨u, ?_, hready, ?_⟩
      · have hc : cur < cur + (n + 1) := by omega
        rw [FP.Structural.range]
        simp only [Nat.one_ne_zero, if_false, hc, if_true, he,
          bind, Option.bind_some]
        have hstop : cur + (n + 1) = (cur + 1) + n := by omega
        simpa only [hstop] using hu
      · change u.mem = iterateMemory update n (cur + 1) (update cur s.mem)
        rw [hmem, hm]

theorem step_constIf {α : Type} [Inhabited α] (M : Algebra α)
    (b : Bool) (yes no : List ComputeStmt) (s : State α) :
    FP.Structural.step M (.ifThenElse (.constBool b) yes no) s =
      if b then FP.Structural.run M yes s else FP.Structural.run M no s := by
  conv_lhs =>
    rw [FP.Structural.step]
    simp only [FP.Structural.evalOp_unfold, bind, Option.bind, Option.pure_def]

theorem run_constIf {α : Type} [Inhabited α] (M : Algebra α)
    (b : Bool) (before yes no : List ComputeStmt) (s : State α) :
    FP.Structural.run M (before ++ [.ifThenElse (.constBool b) yes no]) s =
      FP.Structural.run M (before ++ (if b then yes else no)) s := by
  rw [FP.Structural.run_append, FP.Structural.run_append]
  apply congrArg ((FP.Structural.run M before s).bind)
  funext u
  cases b <;> simp [FP.Structural.run, step_constIf]

theorem range_statement {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (cur stop : Nat) (body : List ComputeStmt) (s : State α) :
    FP.Structural.step M
      (.forRangeDyn idx (.constNat cur) (.constNat stop) (.constNat 1) body) s =
      FP.Structural.range M idx cur stop 1 body s := by
  conv_lhs =>
    rw [FP.Structural.step]
    simp only [FP.Structural.evalOp_unfold, bind, Option.bind, Option.pure_def]

theorem exec_one {α : Type} [Inhabited α] (M : Algebra α)
    (k : ComputeKernel) (before yes no : List ComputeStmt) (s entry last : State α)
    (hsplit : k.surfaceBody = before ++ [.ifThenElse (.constBool Bool.true) yes no])
    (hbefore : FP.Structural.run M before s = some entry)
    (hyes : FP.Structural.run M yes entry = some last) :
    FP.Structural.exec M k s = some last := by
  change FP.Structural.run M k.surfaceBody s = some last
  rw [hsplit, run_constIf]
  simp only [↓reduceIte, FP.Structural.run_append, hbefore, Option.bind_some, hyes]

theorem exec_grid {α : Type} [Inhabited α] (M : Algebra α)
    (k : ComputeKernel) (before yes : List ComputeStmt) (init : ComputeStmt)
    (body : List ComputeStmt) (iterations : Nat) (s entry last : State α)
    (hsplit : k.surfaceBody = before ++ [.ifThenElse (.constBool Bool.false) yes
      [init, .forRangeDyn "j" (.constNat 0) (.constNat iterations) (.constNat 1) body]])
    (hheader : FP.Structural.run M (before ++ [init]) s = some entry)
    (hloop : FP.Structural.range M "j" 0 iterations 1 body entry = some last) :
    FP.Structural.exec M k s = some last := by
  change FP.Structural.run M k.surfaceBody s = some last
  rw [hsplit, run_constIf]
  simp only [Bool.false_eq_true, if_false]
  change FP.Structural.run M (before ++ ([init] ++
    [.forRangeDyn "j" (.constNat 0) (.constNat iterations) (.constNat 1) body])) s = some last
  rw [← List.append_assoc, FP.Structural.run_append, hheader]
  simp only [Option.bind_some, FP.Structural.run, range_statement,
    hloop, bind, Option.bind_some]

#axiomsClean range_memory
#axiomsClean step_constIf
#axiomsClean run_constIf
#axiomsClean range_statement
#axiomsClean exec_one
#axiomsClean exec_grid
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric

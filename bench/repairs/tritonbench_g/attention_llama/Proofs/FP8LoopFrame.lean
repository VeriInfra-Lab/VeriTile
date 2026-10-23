import bench.repairs.tritonbench_g.attention_llama.Proofs.FP8Iteration

namespace VeriTile.Bench.Repairs.AttentionLlama.FP8LoopFrame
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open RealKernel FP8Iteration
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

private def assignsAway (target : RegName) : Stmt → Prop
  | .assign _ _ name _ => name ≠ target
  | _ => False

private theorem assignments_frame (name : RegName) (d : TileDType) (sh : TileShape)
    (code : List Stmt) (hcode : ∀ st ∈ code, assignsAway name st)
    (s t : BlockState) (hrun : stepStmts code s = some t) :
    t.regs d sh name = s.regs d sh name := by
  induction code generalizing s with
  | nil =>
    simp only [stepStmts.nil, Option.some.injEq] at hrun
    subst t
    rfl
  | cons st rest ih =>
    have hc := hcode st (by simp)
    have hrest : ∀ st ∈ rest, assignsAway name st :=
      fun st h => hcode st (by simp [h])
    cases st <;> simp only [assignsAway] at hc
    rename_i dtype shape dest e
    cases he : evalOp e s with
    | none => simp [stepStmts, stepStmt, he] at hrun
    | some tile =>
      have hr : stepStmts rest (s.setReg dest dtype shape tile) = some t := by
        simpa only [stepStmts, stepStmt, he, bind, Option.bind_some] using hrun
      rw [ih hrest _ hr]
      simp [Ne.symm hc]

private def advance (name : RegName) (stride BN BD : Nat) : Stmt :=
  .assign .ptr [BN, BD] name
    (.ptrAdd Broadcast.scalarR (.ref .ptr [BN, BD] name)
      (.mul .nat Broadcast.nil (.constNat BN) (.constNat stride)))

/-- All computation before the two final pointer updates. Taking the final
two statements is independent of how many arithmetic statements precede them;
`split_inline` checks those statements against the source explicitly. -/
private def core (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal : Bool) : List Stmt :=
  (inlineLoop scale skn svk N SP BM BN BD causal).dropLast.dropLast

private theorem split_inline (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal : Bool) :
    inlineLoop scale skn svk N SP BM BN BD causal =
      core scale skn svk N SP BM BN BD causal ++
        [advance "k_ptrs" skn BN BD, advance "v_ptrs" svk BN BD] := by
  cases causal <;> rfl

private theorem core_keeps (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal : Bool) (name : RegName)
    (hname : name ∈ ["offs_n", "k_ptrs", "v_ptrs"]) :
    ∀ st ∈ core scale skn svk N SP BM BN BD causal, assignsAway name st := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hname
  rcases hname with rfl | rfl | rfl
  all_goals
    cases causal <;>
      simp [core, inlineLoop, SourceExecution.loopPrefix, alLoopTailG, alCausalWhereStmt,
        FP8RealSource.lowerLoad, FP8RealSource.byteLoad?, assignsAway]

/-- The successful source iteration preserves offs_n and advances both actual
pointer tiles, including arbitrary strides and causal/candidate branches. -/
theorem registers (K V : RegionName) (s t : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn SN N SP BM BN BD : Nat) (causal : Bool)
    (hkp : s.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hvp : s.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn SN BN BD))
    (hrun : stepStmts (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) s = some t) :
    t.regs .nat [BN] "offs_n" = s.regs .nat [BN] "offs_n" ∧
      t.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk (SN + BN) BN BD) ∧
      t.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn (SN + BN) BN BD) := by
  rw [source_inline, split_inline, ProgramRewrite.append] at hrun
  cases hc : stepStmts (core scale skn svk N SP BM BN BD causal) s with
  | none => simp [hc] at hrun
  | some u =>
    have hk : u.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD) :=
      (assignments_frame "k_ptrs" .ptr [BN, BD] _
        (core_keeps scale skn svk N SP BM BN BD causal "k_ptrs" (by simp)) s u hc).trans hkp
    have hv : u.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn SN BN BD) :=
      (assignments_frame "v_ptrs" .ptr [BN, BD] _
        (core_keeps scale skn svk N SP BM BN BD causal "v_ptrs" (by simp)) s u hc).trans hvp
    have hn := assignments_frame "offs_n" .nat [BN] _
      (core_keeps scale skn svk N SP BM BN BD causal "offs_n" (by simp)) s u hc
    simp only [hc, Option.bind_some] at hrun
    unfold advance at hrun
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (al_ptr_advance_eval u K "k_ptrs" kBase skn skk SN BN BN BD hk))] at hrun
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (al_ptr_advance_eval _ V "v_ptrs" vBase svk svn SN BN BN BD (by simpa using hv))),
      stepStmts.nil, Option.some.injEq] at hrun
    subst t
    simp [hn]

end VeriTile.Bench.Repairs.AttentionLlama.FP8LoopFrame

import bench.tritonbench_g.attn_fwd_triton.Kernels
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Semantics.ProgramRewrite

/- A source-derived view of a constexpr branch with a shared continuation.
The frontend retains different integer register types on its two exits.
Factoring only a literal condition preserves the selected branch, including
its original casts and loop bounds. It never evaluates numerical operations. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.ControlView
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def compute (before setup : Nat) : List ComputeStmt → List ComputeStmt
  | [] => []
  | st::rest => match before with
    | n+1 => st :: compute n setup rest
    | 0 => match st with
      | .ifThenElse (.constBool b) yes no =>
        .ifThenElse (.constBool b) (yes.take setup) [] ::
          (if b then yes.drop setup else no) ++ rest
      | _ => st::rest

def algorithm (before setup : Nat) : List Stmt → List Stmt
  | [] => []
  | st::rest => match before with
    | n+1 => st :: algorithm n setup rest
    | 0 => match st with
      | .ifThenElse (.constBool b) yes no =>
        .ifThenElse (.constBool b) (yes.take setup) [] ::
          (if b then yes.drop setup else no) ++ rest
      | _ => st::rest

inductive ComputeBranch : Nat → List ComputeStmt → Prop
  | here (b yes no rest) : ComputeBranch 0 (.ifThenElse (.constBool b) yes no :: rest)
  | next (st rest n) (h : ComputeBranch n rest) : ComputeBranch (n+1) (st::rest)

inductive AlgorithmBranch : Nat → List Stmt → Prop
  | here (b yes no rest) : AlgorithmBranch 0 (.ifThenElse (.constBool b) yes no :: rest)
  | next (st rest n) (h : AlgorithmBranch n rest) : AlgorithmBranch (n+1) (st::rest)

open FP.Structural in
theorem compute_run {α : Type} [Inhabited α] (M : Algebra α)
    (before setup : Nat) (code : List ComputeStmt) (found : ComputeBranch before code) (s : State α) :
    run M (compute before setup code) s = run M code s := by
  induction found generalizing s with
  | here b yes no rest =>
    cases b
    · simp [compute,run,step,evalOp_unfold,bind,Option.bind,run_append]
    · simp only [compute,↓reduceIte,List.cons_append]
      simp only [run,step,evalOp_unfold,bind,Option.bind,↓reduceIte]
      change (run M (yes.take setup) s).bind (run M (yes.drop setup ++ rest)) =
        (run M yes s).bind (run M rest)
      rw [←run_append,←List.append_assoc,List.take_append_drop,run_append]
  | next st rest n h ih =>
    simp only [compute,run]
    cases step M st s <;> simp only [Option.bind_none,Option.bind_some,ih]

theorem algorithm_run (before setup : Nat) (code : List Stmt)
    (found : AlgorithmBranch before code) (s : BlockState) :
    stepStmts (algorithm before setup code) s = stepStmts code s := by
  induction found generalizing s with
  | here b yes no rest =>
    cases b
    · simp only [algorithm,Bool.false_eq_true,↓reduceIte,List.cons_append,stepStmts,stepStmt,
        evalOp.eq_def,bind,Option.bind,Tile.scalar_data,↓reduceIte]
      rw [ProgramRewrite.append]
      cases stepStmts no s <;> rfl
    · have splitRun := ProgramRewrite.append (yes.take setup) (yes.drop setup ++ rest) s
      rw [←List.append_assoc,List.take_append_drop,ProgramRewrite.append] at splitRun
      simp only [algorithm,↓reduceIte,List.cons_append,stepStmts,stepStmt,
        evalOp.eq_def,bind,Option.bind,Tile.scalar_data,↓reduceIte]
      cases hx : stepStmts (yes.take setup) s <;> cases hy : stepStmts yes s <;>
        simpa only [hx,hy,bind,Option.bind] using splitRun.symm
  | next st rest n h ih =>
    simp only [algorithm,stepStmts]
    cases stepStmt st s <;> simp only [Option.bind_none,Option.bind_some,ih]

open FP.Structural in
theorem compute_safe {α : Type} [Inhabited α] (M : Algebra α)
    (before setup : Nat) (code : List ComputeStmt) (found : ComputeBranch before code)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (compute before setup code) s ↔ Safety.traceSafeList M bounds code s := by
  induction found generalizing s with
  | here b yes no rest =>
    cases b
    · simp [compute,List.cons_append,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
        Safety.safeOp_unfold,step,evalOp_unfold,bind,Option.bind,run,Safety.traceSafeList_append]
    · have key : ∀ a b : List ComputeStmt,
          Safety.traceSafeList M bounds (.ifThenElse (.constBool Bool.true) a [] :: b ++ rest) s ↔
          Safety.traceSafeList M bounds (.ifThenElse (.constBool Bool.true) (a++b) no :: rest) s := by
        intro a b
        simp only [List.cons_append,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
          Safety.safeOp_unfold,step,evalOp_unfold,bind,Option.bind,↓reduceIte]
        simp only [Safety.traceSafeList_append,run_append]
        cases hx : run M a s <;> simp [hx,bind,Option.bind,and_assoc]
      simpa only [compute,↓reduceIte,List.take_append_drop] using key (yes.take setup) (yes.drop setup)
  | next st rest n h ih =>
    simp only [compute,Safety.traceSafeList_cons]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

theorem signed_tail (i n cur : Nat) :
    decide ((i : Int)<(n : Int)-(cur : Int))=decide (i<n-cur) := by
  by_cases h : (i : Int)<(n : Int)-(cur : Int)
  · have hn : i<n-cur := by omega
    simp [h,hn]
  · have hn : ¬ i<n-cur := by omega
    simp [h,hn]

theorem signed_tail_add (i n cur : Nat) :
    decide ((i : Int)<(n : Int)-(cur : Int))=decide (cur+i<n) := by
  by_cases h : (i : Int)<(n : Int)-(cur : Int)
  · have hn : cur+i<n := by omega
    simp [h,hn]
  · have hn : ¬cur+i<n := by omega
    simp [h,hn]

theorem signed_min (x y : Nat) :
    (if (x : Int)<(y : Int) then (x : Int) else (y : Int))=(min x y : Nat) := by
  split_ifs with h
  · rw [Nat.min_eq_left (by omega)]
  · rw [Nat.min_eq_right (by omega)]

theorem signed_div (x b : Nat) :
    (if (x : Int)<0 then -(-(x : Int)/(b : Int)) else (x : Int)/(b : Int))=(x/b : Nat) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte,Int.natCast_ediv]

theorem signed_mod (x b : Nat) :
    (x : Int)-(if (x : Int)<0 then -((b : Int)*(-(x : Int)/(b : Int)))
      else (b : Int)*((x : Int)/(b : Int)))=(x%b : Nat) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte]
  rw [←Int.emod_def,←Int.natCast_emod]

theorem signed_cdiv (x b : Nat) :
    (if (x : Int)+((b : Int)-1)<0 then
        -(-((x : Int)+((b : Int)-1))/(b : Int))
      else ((x : Int)+((b : Int)-1))/(b : Int))=((x+b-1)/b : Nat) := by
  by_cases hz : b=0
  · simp [hz]
  · have hn : 1≤b := by omega
    have ha : (x : Int)+((b : Int)-1)=((x+b-1 : Nat) : Int) := by omega
    rw [ha]
    exact signed_div (x+b-1) b

open FP.Structural in
theorem signed_nat_done {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (cur stop stride : Nat) (body : List ComputeStmt) (s : State α) (h : stop≤cur) :
    rangeInt M idx (cur : Int) (stop : Int) stride body s=some s := by
  rw [rangeInt]
  simp [show ¬(cur : Int)<(stop : Int) by omega]

open FP.Structural in
theorem signed_nat_step {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (cur stop stride : Nat) (body : List ComputeStmt) (s : State α)
    (hd : stride≠0) (h : cur<stop) :
    rangeInt M idx (cur : Int) (stop : Int) stride body s =
      (run M body (s.setReg idx .int [] (fun _=>(cur : Int)))).bind
        (rangeInt M idx ((cur+stride : Nat) : Int) (stop : Int) stride body) := by
  rw [rangeInt]
  simp [hd,show (cur : Int)<(stop : Int) by omega,Int.natCast_add]

theorem algorithm_nat_done (idx : RegName) (cur stop stride : Nat) (body : List Stmt)
    (s : BlockState) (h : stop≤cur) :
    stepForRangeIntAux idx (cur : Int) (stop : Int) stride body s=some s := by
  rw [stepForRangeIntAux]
  simp [show ¬(cur : Int)<(stop : Int) by omega]

theorem algorithm_nat_step (idx : RegName) (cur stop stride : Nat) (body : List Stmt)
    (s : BlockState) (hd : stride≠0) (h : cur<stop) :
    stepForRangeIntAux idx (cur : Int) (stop : Int) stride body s =
      (stepStmts body (s.setReg idx .int [] (Tile.scalar (cur : Int)))).bind
        (stepForRangeIntAux idx ((cur+stride : Nat) : Int) (stop : Int) stride body) := by
  rw [stepForRangeIntAux]
  simp only [hd,show (cur : Int)<(stop : Int) by omega,↓reduceIte,Int.natCast_add]
  cases stepStmts body (s.setReg idx .int [] (Tile.scalar (cur : Int))) <;> rfl

open FP.Structural in
theorem signed_range_invariant {α : Type} [Inhabited α] (M : Algebra α)
    {idx : RegName} {stop : Int} {stride : Nat} {body : List ComputeStmt}
    {P : Int → State α → Prop} (hd : stride ≠ 0)
    (hstep : ∀ i s, i < stop → P i s →
      ∃ t, run M body (s.setReg idx .int [] (fun _ => i))=some t ∧ P (i+stride) t)
    (cur : Int) (s : State α) (hs : P cur s) :
    ∃ last t, rangeInt M idx cur stop stride body s=some t ∧ stop≤last ∧ P last t := by
  have key : ∀ k i s, (stop-i).toNat=k → P i s →
      ∃ last t, rangeInt M idx i stop stride body s=some t ∧ stop≤last ∧ P last t := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
      intro i s hk hs
      by_cases hDone : stop ≤ i
      · exact ⟨i,s,by rw [rangeInt]; simp [hd,not_lt.mpr hDone],hDone,hs⟩
      · have active : i<stop := by omega
        obtain ⟨t,ht,hp⟩ := hstep i s active hs
        obtain ⟨last,u,hu,hlast,hpu⟩ := ih (stop-(i+stride)).toNat (by omega)
          (i+stride) t rfl hp
        refine ⟨last,u,?_,hlast,hpu⟩
        rw [rangeInt]
        simp only [hd,active,↓reduceIte,ht,Option.bind_some]
        exact hu
  exact key (stop-cur).toNat cur s rfl hs

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.ControlView

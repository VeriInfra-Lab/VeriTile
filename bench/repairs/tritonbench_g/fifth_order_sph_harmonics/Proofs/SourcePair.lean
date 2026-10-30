import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Grid
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.SourcePair
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000

/-- Proof-only decomposition of the rounded power assignments. -/
def names : Fin 3 → RegName × RegName × RegName × RegName × RegName
  | ⟨0,_⟩ => ("x","VAR08","VAR07","VAR06","VAR05")
  | ⟨1,_⟩ => ("y","VAR17","VAR16","VAR15","VAR14")
  | ⟨_,_⟩ => ("z","VAR26","VAR25","VAR24","VAR23")
def ref (B : Nat) (name : RegName) : Op .real [B] := .ref .real [B] name
def mul (a b : Op .real [B]) : Op .real [B] := .mul .real (.consSame .nil) a b
def assign (name : RegName) (value : Op .real [B]) : ComputeStmt :=
  .assign .real [B] name (.compute (.alg .fp32 value))
def group (B : Nat) (coordAxis : Fin 3) (fifth shared : Bool) : List ComputeStmt :=
  let ns := names coordAxis
  let x := ref B ns.1
  let a2 := ns.2.1
  let a3 := ns.2.2.1
  let a4 := ns.2.2.2.1
  let a5 := ns.2.2.2.2
  if shared then
    [assign a2 (mul x x),assign a3 (mul (ref B a2) x),assign a4 (mul (ref B a3) x)] ++
      (if fifth then [assign a5 (mul (ref B a4) x)] else [])
  else (if fifth then [assign a5 (mul (mul (mul (mul x x) x) x) x)] else []) ++
    [assign a4 (mul (mul (mul x x) x) x),assign a3 (mul (mul x x) x),assign a2 (mul x x)]

def powers (B : Nat) (d : Grid.Direction) (shared : Bool) :=
  group B 0 (match d with | .forward => Bool.true | .backward => Bool.false) shared ++
  group B 1 (match d with | .forward => Bool.true | .backward => Bool.false) shared ++
  group B 2 (match d with | .forward => Bool.true | .backward => Bool.false) shared

def original (d : Grid.Direction) (p : Parameters) :=
  match d with | .forward => forwardKernel p | .backward => backwardKernel p
def candidate (d : Grid.Direction) (p : Parameters) :=
  match d with | .forward => forwardPowersKernel p | .backward => backwardPowersKernel p
def source (d : Grid.Direction) (shared : Bool) (p : Parameters) :=
  if shared then candidate d p else original d p

def before : Grid.Direction → Nat | .forward => 37 | .backward => 104
def after : Grid.Direction → Nat | .forward => 49 | .backward => 113

theorem split (d : Grid.Direction) (shared : Bool) (p : Parameters) :
    (source d shared p).surfaceBody = (original d p).surfaceBody.take (before d) ++
      powers p.blockSize d shared ++ (original d p).surfaceBody.drop (after d) := by
  cases d <;> cases shared <;> rfl

/-- Assignment sharing preserves exactly the same rounded operation tree.
The only reordered effects are writes to distinct local registers. -/
theorem group_exec {α : Type} [Inhabited α] (M : Algebra α) (B : Nat)
    (coordAxis : Fin 3) (fifth : Bool) (s : State α) :
    run M (group B coordAxis fifth Bool.false) s = run M (group B coordAxis fifth Bool.true) s := by
  fin_cases coordAxis
  · cases fifth
    all_goals
      cases hv : s.regs .real [B] "x" <;>
        simp [group,names,ref,mul,assign,run,step,evalExpr,evalComputeOp,evalOp_unfold,
          numeric,ComputeDType.eraseDType,State.setReg,hv]
    all_goals
      funext dtype shape name
      split_ifs <;> simp_all
  · cases fifth
    all_goals
      cases hv : s.regs .real [B] "y" <;>
        simp [group,names,ref,mul,assign,run,step,evalExpr,evalComputeOp,evalOp_unfold,
          numeric,ComputeDType.eraseDType,State.setReg,hv]
    all_goals
      funext dtype shape name
      split_ifs <;> simp_all
  · cases fifth
    all_goals
      cases hv : s.regs .real [B] "z" <;>
        simp [group,names,ref,mul,assign,run,step,evalExpr,evalComputeOp,evalOp_unfold,
          numeric,ComputeDType.eraseDType,State.setReg,hv]
    all_goals
      funext dtype shape name
      split_ifs <;> simp_all

theorem powers_exec {α : Type} [Inhabited α] (M : Algebra α) (B : Nat)
    (d : Grid.Direction) : run M (powers B d Bool.false) = run M (powers B d Bool.true) := by
  funext s
  simp only [powers,run_append,group_exec]

theorem group_safe {α : Type} [Inhabited α] (M : Algebra α) (B : Nat)
    (coordAxis : Fin 3) (fifth shared : Bool) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (group B coordAxis fifth shared) s := by
  fin_cases coordAxis <;> cases fifth <;> cases shared <;>
    simp [group,names,assign,ref,mul,Safety.traceSafeList_cons,Safety.stepSafe,
      Safety.safeExpr,Safety.safeCompute,Safety.safeOp_unfold]

theorem powers_safe {α : Type} [Inhabited α] (M : Algebra α) (B : Nat)
    (d : Grid.Direction) (shared : Bool) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (powers B d shared) s := by
  simp only [powers,Safety.traceSafeList_append,group_safe,true_and,implies_true,forall_const]

theorem source_exec {α : Type} [Inhabited α] (M : Algebra α) (d : Grid.Direction)
    (p : Parameters) (s : State α) : exec M (original d p) s = exec M (candidate d p) s := by
  change run M (source d Bool.false p).surfaceBody s = run M (source d Bool.true p).surfaceBody s
  simp only [split,run_append,powers_exec]

theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (d : Grid.Direction)
    (p : Parameters) (s : State α) (bounds : RegionName → Nat) :
    TraceSafe M bounds (original d p) s ↔ TraceSafe M bounds (candidate d p) s := by
  change Safety.traceSafeList M bounds (source d Bool.false p).surfaceBody s ↔
    Safety.traceSafeList M bounds (source d Bool.true p).surfaceBody s
  simp only [split,Safety.traceSafeList_append,run_append,powers_exec,powers_safe]

end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.SourcePair

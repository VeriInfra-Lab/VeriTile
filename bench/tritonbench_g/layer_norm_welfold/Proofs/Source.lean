import bench.tritonbench_g.layer_norm_welfold.Kernels
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeHost
import Lean

/-! Proof fragments project the one literal repaired source. A fragment that
ignores the ordered tree is normalized before the unused formal is removed;
its source decomposition is checked again by definitional equality. No second
quoted implementation or numerical evaluation is kept here. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural
namespace Reuse
export VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
  (ReductionTree withEquality)
export VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
  (blockCount block_count_covers block_count_positive column_chunk chunk_unique
   row_column_injective output_bounds statistics_reinterpret Separation rowMiss
   typed_scatter_memory_congr typed_scatter_regs typed_scatter_undef
   atRow runRows Tensor Allocations StorageDType InputContract Returned
   emptyStrided reinterpret outputTensor inverseBacking inverseTensor)
end Reuse
open Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

namespace SourceBinding
def tree {B : Nat} : Reuse.ReductionTree B → SourceCode.ReductionTree B
  | .leaf i => .leaf i
  | .node l r => .node (tree l) (tree r)
end SourceBinding

open Lean Elab Term Meta in
private partial def usesFormal (id : FVarId) : Expr → Bool
  | .fvar found => id == found
  | .app f a => usesFormal id f || usesFormal id a
  | .lam _ t b _ | .forallE _ t b _ => usesFormal id t || usesFormal id b
  | .letE _ t v b _ => usesFormal id t || usesFormal id v || usesFormal id b
  | .mdata _ b | .proj _ _ b => usesFormal id b
  | _ => Bool.false

open Lean Elab Term Meta in
private partial def fragmentSpine (value : Expr) : MetaM Expr := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return value
  if value.isAppOfArity ``List.cons 3 then
    let args := value.getAppArgs
    return mkAppN value.getAppFn #[args[0]!,args[1]!,← fragmentSpine args[2]!]
  throwError "source fragment must expose a finite statement list"

open Lean Elab Term Meta in
elab "source_fragment " source:term : term => do
  let source ← elabTerm source none
  synthesizeSyntheticMVarsNoPostponing
  let source ← instantiateMVars source
  let type ← whnf (← inferType source)
  unless type.isForall do throwError "source fragment requires a tree formal"
  withLocalDeclD `sourceTree type.bindingDomain! fun tree => do
    let result ← instantiateMVars (← fragmentSpine (mkApp source tree))
    if usesFormal tree.fvarId! result then throwError "source fragment still depends on its tree formal"
    return result

open Lean Elab Term Meta in
elab "source_statements " source:term : term => do
  let type ← mkAppM ``List #[mkConst ``ComputeStmt]
  let source ← elabTermEnsuringType source type
  synthesizeSyntheticMVarsNoPostponing
  fragmentSpine (← instantiateMVars source)

def code (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) : List ComputeStmt :=
  (SourceCode.rowKernel X W Bias Y Mean Rstd S D B (SourceBinding.tree t)).surfaceBody

def init (S B : Nat) : List ComputeStmt :=
  source_fragment (fun (t : SourceCode.ReductionTree B) =>
    (SourceCode.rowKernel "X" "W" "Bias" "Y" "Mean" "Rstd" S 0 B t).surfaceBody.take 6)

def reset (B : Nat) : List ComputeStmt :=
  source_fragment (fun (t : SourceCode.ReductionTree B) =>
    ((SourceCode.rowKernel "X" "W" "Bias" "Y" "Mean" "Rstd" 0 0 B t).surfaceBody.drop 12).take 1)

def phaseCode (phase : Bool) (X : RegionName) (D B : Nat) : List ComputeStmt :=
  if phase then
    source_fragment (fun (t : SourceCode.ReductionTree B) =>
      match (SourceCode.rowKernel X "W" "Bias" "Y" "Mean" "Rstd" 0 D B t).surfaceBody.drop 13 with
      | .forRange _ _ _ _ body :: _ => body
      | _ => [])
  else
    source_fragment (fun (t : SourceCode.ReductionTree B) =>
      match (SourceCode.rowKernel X "W" "Bias" "Y" "Mean" "Rstd" 0 D B t).surfaceBody.drop 6 with
      | .forRange _ _ _ _ body :: _ => body
      | _ => [])

/-- One iteration at the actual dynamic-range index k*B. -/
def chunk (phase : Bool) (X : RegionName) (D B k : Nat) : List ComputeStmt :=
  [.assign .nat [] "roffset" (.alg (.constNat (k*B)))] ++ phaseCode phase X D B

def chunks (phase : Bool) (X : RegionName) (D B : Nat) : Nat → Nat → List ComputeStmt
  | _,0 => []
  | start,count+1 => chunk phase X D B start ++ chunks phase X D B (start+1) count

def sumExpr {B : Nat} (phase : Bool) : ReductionTree B → Op .real []
  | .leaf i => .remap (shape := [1,B]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩)
      (.ref .real [1,B] (if phase then "_tmp12" else "_tmp3"))
  | .node l r => .add .real .nil (sumExpr phase l) (sumExpr phase r)

def sumCode {B : Nat} (phase : Bool) (tree : ReductionTree B) : List ComputeStmt :=
  if phase then
    source_statements (((code "X" "W" "Bias" "Y" "Mean" "Rstd" 0 0 B tree).drop 14).take 1)
  else
    source_statements (((code "X" "W" "Bias" "Y" "Mean" "Rstd" 0 0 B tree).drop 7).take 1)

/-- The source keeps tmp6 through its second and third loops. -/
def meanCode (Mean : RegionName) (D : Nat) : List ComputeStmt :=
  source_fragment (fun (t : SourceCode.ReductionTree 0) =>
    ((SourceCode.rowKernel "X" "W" "Bias" "Y" Mean "Rstd" 0 D 0 t).surfaceBody.drop 8).take 4)

def inverseCode (Rstd : RegionName) (D : Nat) : List ComputeStmt :=
  source_fragment (fun (t : SourceCode.ReductionTree 0) =>
    ((SourceCode.rowKernel "X" "W" "Bias" "Y" "Mean" Rstd 0 D 0 t).surfaceBody.drop 15).take 7)

theorem source_init (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :
    (code X W Bias Y Mean Rstd S D B t).take 6 = init S B := rfl

theorem source_reset (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :
    ((code X W Bias Y Mean Rstd S D B t).drop 12).take 1 = reset B := rfl

theorem source_mean (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :
    ((code X W Bias Y Mean Rstd S D B t).drop 8).take 4 = meanCode Mean D := rfl

theorem source_inverse (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :
    ((code X W Bias Y Mean Rstd S D B t).drop 15).take 7 = inverseCode Rstd D := rfl

theorem source_phases (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B)
    (phase : Bool) :
    ((code X W Bias Y Mean Rstd S D B t).drop (if phase then 13 else 6)).head? =
      some (.forRange "roffset" 0 D B (phaseCode phase X D B)) := by
  cases phase <;> rfl

example (X : RegionName) (D B : Nat) : (phaseCode Bool.false X D B).length = 9 := rfl
example (X : RegionName) (D B : Nat) : (phaseCode Bool.true X D B).length = 11 := rfl

#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.LayerNormWelfold

import bench.tritonbench_g.fused_layernorm_triton.Kernels
import VeriTile.Triton.Source

/-! Proof-only projections of the complete readable kernels. No projected list
is an authoritative source definition. Every context equation is checked by
reduction against the full quotation. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000

def source (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : ComputeKernel :=
  if hoist then Kernels.hoisted X W Bias Y Mean Rstd S D B tree
  else Kernels.kernel X W Bias Y Mean Rstd S D B tree

def statistics (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : Source.IntRange :=
  (Source.intRange? (source hoist X W Bias Y Mean Rstd S D B tree).surfaceBody "roffset").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

def normalization (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : Source.IntRange :=
  (Source.intRange? (statistics hoist X W Bias Y Mean Rstd S D B tree).after "roffset").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem complete (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (source hoist X W Bias Y Mean Rstd S D B tree).surfaceBody =
      (statistics hoist X W Bias Y Mean Rstd S D B tree).before ++
      [.forRangeInt "roffset" (statistics hoist X W Bias Y Mean Rstd S D B tree).start
        (statistics hoist X W Bias Y Mean Rstd S D B tree).stop
        (statistics hoist X W Bias Y Mean Rstd S D B tree).stride
        (statistics hoist X W Bias Y Mean Rstd S D B tree).body] ++
      (normalization hoist X W Bias Y Mean Rstd S D B tree).before ++
      [.forRangeInt "roffset" (normalization hoist X W Bias Y Mean Rstd S D B tree).start
        (normalization hoist X W Bias Y Mean Rstd S D B tree).stop
        (normalization hoist X W Bias Y Mean Rstd S D B tree).stride
        (normalization hoist X W Bias Y Mean Rstd S D B tree).body] := by
  cases hoist <;> rfl

theorem common_prefix (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (statistics Bool.true X W Bias Y Mean Rstd S D B tree).before =
      (statistics Bool.false X W Bias Y Mean Rstd S D B tree).before := by rfl

theorem common_body (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (statistics Bool.true X W Bias Y Mean Rstd S D B tree).body =
      (statistics Bool.false X W Bias Y Mean Rstd S D B tree).body := by rfl

def projections (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : List ComputeStmt :=
  (normalization Bool.true X W Bias Y Mean Rstd S D B tree).before.drop
    (normalization Bool.false X W Bias Y Mean Rstd S D B tree).before.length

theorem hoisted_before (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (normalization Bool.true X W Bias Y Mean Rstd S D B tree).before =
      (normalization Bool.false X W Bias Y Mean Rstd S D B tree).before ++
      projections X W Bias Y Mean Rstd S D B tree := by rfl

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
    return mkAppN value.getAppFn #[args[0]!,args[1]!,←fragmentSpine args[2]!]
  throwError "source fragment must expose a finite statement list"

/- Discard a tree formal only after reducing the actual source projection
and checking that no remaining instruction or type uses that formal. -/
open Lean Elab Term Meta in
elab "kernel_source_fragment " source:term : term => do
  let source ← elabTerm source none
  synthesizeSyntheticMVarsNoPostponing
  let source ← instantiateMVars source
  let type ← whnf (←inferType source)
  unless type.isForall do throwError "source fragment requires a tree formal"
  withLocalDeclD `sourceTree type.bindingDomain! fun tree => do
    let result ← instantiateMVars (←fragmentSpine (mkApp source tree))
    if usesFormal tree.fvarId! result then throwError "source fragment still depends on its tree formal"
    return result

def normalizationBody (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) : List ComputeStmt :=
  match hoist with
  | Bool.false => kernel_source_fragment (fun (tree : Kernels.ReductionTree B) =>
      (normalization Bool.false X W Bias Y Mean Rstd S D B tree).body)
  | Bool.true => kernel_source_fragment (fun (tree : Kernels.ReductionTree B) =>
      (normalization Bool.true X W Bias Y Mean Rstd S D B tree).body)

def projectionBody (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) : List ComputeStmt :=
  kernel_source_fragment (fun (tree : Kernels.ReductionTree B) =>
    projections X W Bias Y Mean Rstd S D B tree)

theorem normalizationBody_eq (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    normalizationBody hoist X W Bias Y Mean Rstd S D B =
      (normalization hoist X W Bias Y Mean Rstd S D B tree).body := by cases hoist <;> rfl

theorem projectionBody_eq (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    projectionBody X W Bias Y Mean Rstd S D B = projections X W Bias Y Mean Rstd S D B tree := rfl

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces

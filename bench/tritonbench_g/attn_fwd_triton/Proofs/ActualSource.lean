import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedExecution
import bench.tritonbench_g.attn_fwd_triton.Proofs.Real
import bench.tritonbench_g.attn_fwd_triton.Proofs.SourceFragments

/-! Real proof views are projections of complete source-derived statement
blocks. Successful projection is required before distributing over blocks;
failed compute projection is never silently treated as an empty program. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 4000000

abbrev Parts := List (List ComputeStmt)

def project (code : List ComputeStmt) : List Stmt :=
  Stmt.eraseDTypeList (ComputeKernel.mk [] [] code).toAlgKernel.body

def Projectable (code : List ComputeStmt) : Prop :=
  ∃ algorithm, ComputeStmt.listToAlgorithm? code=Except.ok algorithm

theorem erase_append (left right : List Stmt) :
    Stmt.eraseDTypeList (left++right)=Stmt.eraseDTypeList left++Stmt.eraseDTypeList right := by
  induction left with
  | nil=>simp [project,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?,Stmt.eraseDTypeList,Except.bind,bind]
  | cons head tail ih=>simp only [List.cons_append,Stmt.eraseDTypeList,ih]

theorem append_projectable {left right : List ComputeStmt}
    (success : Projectable (left++right)) : Projectable left ∧ Projectable right := by
  rcases success with ⟨algorithm,success⟩
  rw [ComputeStmt.listToAlgorithm?_append] at success
  cases hl : ComputeStmt.listToAlgorithm? left with
  | error error=>simp [hl] at success
  | ok l=>
    cases hr : ComputeStmt.listToAlgorithm? right with
    | error error=>simp [hl,hr] at success
    | ok r=>exact ⟨⟨l,hl⟩,⟨r,hr⟩⟩

theorem project_append (left right : List ComputeStmt)
    (success : Projectable (left++right)) :
    project (left++right)=project left++project right := by
  obtain ⟨⟨l,hl⟩,⟨r,hr⟩⟩ := append_projectable success
  simp [project,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?_append,hl,hr,erase_append,Except.bind,bind]

theorem project_flatten (parts : Parts) (success : Projectable parts.flatten) :
    project parts.flatten=(parts.map project).flatten := by
  induction parts with
  | nil=>simp [project,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?,Stmt.eraseDTypeList,Except.bind,bind]
  | cons head rest ih=>
    simp only [List.flatten_cons] at success ⊢
    rw [project_append head rest.flatten success]
    rw [ih (append_projectable success).2]
    rfl

/-- This adapter only transports the established caller binding fields. -/
def config (p : Parameters) : TypedExecution.Config :=
  ⟨p.Q,p.K,p.V,p.QS,p.KS,p.Out,
   p.queryBatch,p.queryHead,p.queryRow,p.queryDim,
   p.keyBatch,p.keyHead,p.keyRow,p.keyDim,
   p.valueBatch,p.valueHead,p.valueRow,p.valueDim,
   p.outputBatch,p.outputHead,p.outputRow,p.outputDim,
   p.batches,p.heads,p.context,p.headDim,p.stage,p.validStage⟩

theorem source_eq (p : Parameters) (BD : Nat) :
    TypedExecution.source (config p) BD Bool.false=source 128 64 BD p Bool.false := rfl

def chunks (p : Parameters) (BD : Nat) : Parts :=
  [TypedExecution.coordinateCode (config p) BD,
   TypedExecution.pointerCode (config p) BD,
   TypedExecution.initializationCode (config p) BD,
   TypedExecution.setupCode (config p) BD Bool.false,
   TypedExecution.loopCode (config p) BD Bool.false,
   TypedExecution.setupCode (config p) BD Bool.true,
   TypedExecution.loopCode (config p) BD Bool.true,
   TypedExecution.tail (config p) BD Bool.false]

theorem typed_assembled (p : Parameters) (BD : Nat) :
    TypedExecution.code (config p) BD=(chunks p BD).flatten := by
  simpa only [chunks,List.flatten_cons,List.flatten_nil,List.append_nil,List.append_assoc] using
    TypedExecution.assembled (config p) BD Bool.false

theorem source_projectable (p : Parameters) (BD : Nat) :
    Projectable (TypedExecution.code (config p) BD) := by
  by_cases hd : p.stage=2
  all_goals simp only [TypedExecution.code,TypedExecution.source,config,
    TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals exact ⟨_,rfl⟩

theorem whole_project (p : Parameters) (BD : Nat) :
    code 128 64 BD p Bool.false=project (TypedExecution.code (config p) BD) := by
  obtain ⟨algorithm,success⟩ := source_projectable p BD
  change (source 128 64 BD p Bool.false).eraseDType.toAlgKernel.body=_
  rw [←source_eq]
  cases h : TypedExecution.source (config p) BD Bool.false with
  | mk inputs outputs body=>
    change ComputeStmt.listToAlgorithm? (TypedExecution.source (config p) BD Bool.false).surfaceBody=Except.ok algorithm at success
    rw [h] at success
    change ComputeStmt.listToAlgorithm? body=Except.ok algorithm at success
    simp [project,TypedExecution.code,h,ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,success,Kernel.eraseDType,ComputeKernel.fromKernelBody,Except.bind,bind]

theorem assembled (p : Parameters) (BD : Nat) :
    code 128 64 BD p Bool.false=((chunks p BD).map project).flatten := by
  rw [whole_project,typed_assembled]
  exact project_flatten _ (by rw [←typed_assembled]; exact source_projectable p BD)

#axiomsClean assembled
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualSource

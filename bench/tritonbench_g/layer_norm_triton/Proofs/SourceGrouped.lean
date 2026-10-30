import bench.tritonbench_g.layer_norm_triton.Proofs.SourceFinal

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceGrouped
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

@[simp] theorem masked_ptr_regs {α ι : Type} (address : ι → RegionName × Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write (address i).1 (address i).2 (value i) else t) s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split <;> exact ih _

def code (stride M N G B : Nat) : List ComputeStmt :=
  (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B).surfaceBody
def prelude (stride M N G B : Nat) := (code stride M N G B).take 7
def body (stride M N G B : Nat) : List ComputeStmt :=
  match (code stride M N G B).drop 7 with
  | (.forRangeDyn _ _ _ _ body)::_ => body
  | _ => []
def suffix (stride M N G B : Nat) := (code stride M N G B).drop 8

theorem code_split (stride M N G B : Nat) : code stride M N G B=
    prelude stride M N G B ++
      [.forRangeDyn "row" (.ref .nat [] "group") (.constNat M) (.constNat G) (body stride M N G B)] ++
      suffix stride M N G B := by rfl

def Ready (B : Nat) (s : State α) : Prop :=
  (∃ v,s.regs .nat [] "group"=some v) ∧
  (∃ v,s.regs .nat [B] "cols"=some v) ∧
  (∃ v,s.regs .bool [B] "mask"=some v) ∧
  (∃ v,s.regs .ptr [B] "DW"=some v) ∧
  (∃ v,s.regs .ptr [B] "DB"=some v) ∧
  (∃ v,s.regs .real [B] "dw_acc"=some v) ∧
  (∃ v,s.regs .real [B] "db_acc"=some v)

theorem prelude_success [Inhabited α] (m : Model α) (stride M N G B : Nat) (s : State α) :
    ∃ t,run (engine m) (prelude stride M N G B) s=some t ∧ Ready B t := by
  simp [prelude,code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Ready,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat]

theorem body_success [Inhabited α] (m : Model α) (stride M N G B i : Nat)
    (s : State α) (h : Ready B s) :
    ∃ t,run (engine m) (body stride M N G B) (s.setReg "row" .nat [] (fun _ => i))=some t ∧ Ready B t := by
  obtain ⟨⟨group,hgroup⟩,⟨cols,hcols⟩,⟨mask,hmask⟩,⟨dwp,hdwp⟩,⟨dbp,hdbp⟩,⟨dw,hdw⟩,⟨db,hdb⟩⟩ := h
  by_cases hi : i=group PUnit.unit <;>
    simp [body,code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Ready,
      engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
      State.setReg,FP.Structural.store,ComputeDType.eraseDType,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb,hi]

theorem suffix_success [Inhabited α] (m : Model α) (stride M N G B : Nat)
    (s : State α) (h : Ready B s) :
    ∃ t,run (engine m) (suffix stride M N G B) s=some t := by
  obtain ⟨⟨group,hgroup⟩,⟨cols,hcols⟩,⟨mask,hmask⟩,⟨dwp,hdwp⟩,⟨dbp,hdbp⟩,⟨dw,hdw⟩,⟨db,hdb⟩⟩ := h
  simp [suffix,code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb]

theorem source_executes [Inhabited α] (m : Model α) (stride M N G B : Nat) (s : State α) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) s=some t := by
  obtain ⟨a,ha,hready⟩ := prelude_success m stride M N G B s
  obtain ⟨group,hgroup⟩ := hready.1
  obtain ⟨b,hb,hbready⟩ := SourceFinal.range_success (engine m) "row" (group PUnit.unit) M G
    (body stride M N G B) (Ready B) (body_success m stride M N G B) a hready
  obtain ⟨t,ht⟩ := suffix_success m stride M N G B b hbready
  refine ⟨t,?_⟩
  change run (engine m) (code stride M N G B) s=some t
  rw [code_split,run_append,run_append,ha]
  simpa only [bind,Option.bind,run,step,evalOp_unfold,hgroup,hb] using ht

#axiomsClean source_executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceGrouped

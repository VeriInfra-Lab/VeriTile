import bench.tritonbench_g.layer_norm_triton.Proofs.SourceModel

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceFinal
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

@[simp] theorem masked_regs {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).regs=s.regs := by
  rw [State.masked_scatter_eq_filter,State.scatter_regs]

def code (M N BM BN : Nat) : List ComputeStmt :=
  (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN).surfaceBody
def prelude (M N BM BN : Nat) := (code M N BM BN).take 4
def body (M N BM BN : Nat) : List ComputeStmt :=
  match (code M N BM BN).drop 4 with
  | (ComputeStmt.forRange _ _ _ _ body)::_ => body
  | _ => []
def suffix (M N BM BN : Nat) := (code M N BM BN).drop 5

theorem code_split (M N BM BN : Nat) : code M N BM BN=
    prelude M N BM BN ++ [.forRange "i" 0 M BM (body M N BM BN)] ++ suffix M N BM BN := by
  rfl

def Ready (BM BN : Nat) (s : State α) : Prop :=
  (∃ v,s.regs .nat [] "pid"=some v) ∧
  (∃ v,s.regs .nat [BN] "cols"=some v) ∧
  (∃ v,s.regs .real [BM,BN] "dw"=some v) ∧
  (∃ v,s.regs .real [BM,BN] "db"=some v)

theorem prelude_success [Inhabited α] (m : Model α) (M N BM BN : Nat) (s : State α) :
    ∃ t,run (engine m) (prelude M N BM BN) s=some t ∧ Ready BM BN t := by
  simp [prelude,code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Ready,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat]

theorem body_success [Inhabited α] (m : Model α) (M N BM BN i : Nat)
    (s : State α) (h : Ready BM BN s) :
    ∃ t,run (engine m) (body M N BM BN) (s.setReg "i" .nat [] (fun _ => i))=some t ∧ Ready BM BN t := by
  obtain ⟨⟨pid,hpid⟩,⟨cols,hcols⟩,⟨dw,hdw⟩,⟨db,hdb⟩⟩ := h
  simp [body,code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Ready,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hpid,hcols,hdw,hdb]

theorem range_success [Inhabited α] (m : Algebra α) (idx : RegName) (cur stop stride : Nat)
    (body : List ComputeStmt) (P : State α → Prop)
    (next : ∀ i s,P s → ∃ t,run m body (s.setReg idx .nat [] (fun _ => i))=some t ∧ P t)
    (s : State α) (hs : P s) : ∃ t,FP.Structural.range m idx cur stop stride body s=some t ∧ P t := by
  rw [FP.Structural.range]
  split
  · exact ⟨s,rfl,hs⟩
  · split
    · obtain ⟨t,ht,hp⟩ := next cur s hs
      simp only [ht,bind,Option.bind]
      exact range_success m idx (cur+stride) stop stride body P next t hp
    · exact ⟨s,rfl,hs⟩
termination_by stop-cur
decreasing_by omega

theorem suffix_success [Inhabited α] (m : Model α) (M N BM BN : Nat)
    (s : State α) (h : Ready BM BN s) :
    ∃ t,run (engine m) (suffix M N BM BN) s=some t := by
  obtain ⟨⟨pid,hpid⟩,⟨cols,hcols⟩,⟨dw,hdw⟩,⟨db,hdb⟩⟩ := h
  simp [suffix,code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hpid,hcols,hdw,hdb]

theorem source_executes [Inhabited α] (m : Model α) (M N BM BN : Nat) (s : State α) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) s=some t := by
  obtain ⟨a,ha,hready⟩ := prelude_success m M N BM BN s
  obtain ⟨b,hb,hbready⟩ := range_success (engine m) "i" 0 M BM (body M N BM BN)
    (Ready BM BN) (body_success m M N BM BN) a hready
  obtain ⟨t,ht⟩ := suffix_success m M N BM BN b hbready
  refine ⟨t,?_⟩
  change run (engine m) (code M N BM BN) s=some t
  rw [code_split,run_append,run_append,ha]
  simpa only [bind,Option.bind,run,step,hb] using ht

#axiomsClean source_executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceFinal

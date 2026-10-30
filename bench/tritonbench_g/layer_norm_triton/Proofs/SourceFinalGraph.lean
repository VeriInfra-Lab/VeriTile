import bench.tritonbench_g.layer_norm_triton.Proofs.SourceFinal
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardGraph

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceFinalGraph
open VeriTile.Triton FP.Structural SourceModel
open SourceForwardGraph (lit bin cast fold_memory_congr)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev Tile (α : Type) (BM BN : Nat) := Values α .real [BM,BN]
abbrev Carries (α : Type) (BM BN : Nat) := Tile α BM BN × Tile α BM BN

def columns (s : State α) (BN : Nat) : Values α .nat [BN] :=
  fun i => s.pids 0*BN+i.1.val

def zero (m : Model α) (BM BN : Nat) : Carries α BM BN :=
  (fun _ => lit m 0,fun _ => lit m 0)

def loadTile [Inhabited α] (m : Model α) (s : State α) (r : RegionName)
    (M N BM BN cur : Nat) : Tile α BM BN := fun i =>
  if cur+i.1.val<M ∧ s.pids 0*BN+i.2.1.val<N then
    (s.mem r ((cur+i.1.val)*N+(s.pids 0*BN+i.2.1.val))).read .real else lit m 0

def next [Inhabited α] (m : Model α) (s : State α) (M N BM BN cur : Nat)
    (a : Carries α BM BN) : Carries α BM BN :=
  (fun i => bin m .add (a.1 i) (loadTile m s "DW" M N BM BN cur i),
   fun i => bin m .add (a.2 i) (loadTile m s "DB" M N BM BN cur i))

def fold [Inhabited α] (m : Model α) (s : State α) (M N BM BN cur : Nat)
    (a : Carries α BM BN) : Carries α BM BN :=
  if BM=0 then a else if cur<M then fold m s M N BM BN (cur+BM) (next m s M N BM BN cur a) else a
termination_by M-cur
decreasing_by omega

def Invariant (s : State α) (BM BN : Nat) (a : Carries α BM BN) (t : State α) : Prop :=
  t.mem=s.mem ∧ t.regs .nat [BN] "cols"=some (columns s BN) ∧
  t.regs .real [BM,BN] "dw"=some a.1 ∧ t.regs .real [BM,BN] "db"=some a.2

theorem prelude_refines [Inhabited α] (m : Model α) (s : State α) (M N BM BN : Nat) :
    ∃ t,run (engine m) (SourceFinal.prelude M N BM BN) s=some t ∧ Invariant s BM BN (zero m BM BN) t := by
  simp [SourceFinal.prelude,SourceFinal.code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Invariant,columns,zero,lit,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat]
  rfl

theorem body_refines [Inhabited α] (m : Model α) (s t : State α) (M N BM BN cur : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t) :
    ∃ u,run (engine m) (SourceFinal.body M N BM BN) (t.setReg "i" .nat [] (fun _ => cur))=some u ∧
      Invariant s BM BN (next m s M N BM BN cur a) u := by
  obtain ⟨hm,hcols,hdw,hdb⟩ := h
  simp [SourceFinal.body,SourceFinal.code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Invariant,columns,next,loadTile,lit,bin,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hcols,hdw,hdb,hm,
    show (0.0 : ℝ)=0 by norm_num]
  constructor <;> rfl

theorem range_refines [Inhabited α] (m : Model α) (s t : State α) (M N BM BN cur : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t) :
    ∃ u,FP.Structural.range (engine m) "i" cur M BM (SourceFinal.body M N BM BN) t=some u ∧
      Invariant s BM BN (fold m s M N BM BN cur a) u := by
  rw [FP.Structural.range,fold]
  split
  · exact ⟨t,rfl,h⟩
  · split
    · obtain ⟨u,hu,hnext⟩ := body_refines m s t M N BM BN cur a h
      simp only [hu,bind,Option.bind]
      exact range_refines m s u M N BM BN (cur+BM) (next m s M N BM BN cur a) hnext
    · exact ⟨t,rfl,h⟩
termination_by M-cur
decreasing_by omega

def reduced (m : Model α) (BM BN : Nat) (tile : Tile α BM BN) : Values α .real [BN] :=
  m.scalar.reduceSum (some .fp32) (shape:=[BM,BN]) ⟨0,by simp⟩ Bool.false tile

def stores (m : Model α) (s : State α) (N BM BN : Nat) (a : Carries α BM BN) : State α :=
  let w := (TileShape.allIndices [BN]).foldl (fun t i =>
    if columns s BN i<N then t.write "FINAL_DW" (columns s BN i)
      (.mk .real (cast m (reduced m BM BN a.1 i))) else t) s
  (TileShape.allIndices [BN]).foldl (fun t i =>
    if columns s BN i<N then t.write "FINAL_DB" (columns s BN i)
      (.mk .real (cast m (reduced m BM BN a.2 i))) else t) w

theorem suffix_refines [Inhabited α] (m : Model α) (s t : State α) (M N BM BN : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t) :
    ∃ u,run (engine m) (SourceFinal.suffix M N BM BN) t=some u ∧
      u.mem=(stores m s N BM BN a).mem := by
  obtain ⟨hm,hcols,hdw,hdb⟩ := h
  simp [SourceFinal.suffix,SourceFinal.code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hcols,hdw,hdb]
  simp only [stores,reduced,SourceForwardGraph.cast]
  apply fold_memory_congr
  · apply fold_memory_congr
    · exact hm
    · intro u v i huv
      by_cases hi : columns s BN i<N <;> simp [hi,State.write,huv] <;> rfl
  · intro u v i huv
    by_cases hi : columns s BN i<N <;> simp [hi,State.write,huv] <;> rfl

def final [Inhabited α] (m : Model α) (s : State α) (M N BM BN : Nat) : State α :=
  stores m s N BM BN (fold m s M N BM BN 0 (zero m BM BN))

theorem source_refines [Inhabited α] (m : Model α) (s : State α) (M N BM BN : Nat) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) s=some t ∧
      t.mem=(final m s M N BM BN).mem := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s M N BM BN
  obtain ⟨b,hb,hbready⟩ := range_refines m s a M N BM BN 0 (zero m BM BN) hready
  obtain ⟨t,ht,hmem⟩ := suffix_refines m s b M N BM BN _ hbready
  refine ⟨t,?_,hmem⟩
  change run (engine m) (SourceFinal.code M N BM BN) s=some t
  rw [SourceFinal.code_split,run_append,run_append,ha]
  simpa only [bind,Option.bind,run,step,hb] using ht

#axiomsClean range_refines
#axiomsClean source_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceFinalGraph

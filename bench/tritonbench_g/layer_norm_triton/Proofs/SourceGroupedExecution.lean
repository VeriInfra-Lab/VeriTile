import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedGraph

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedExecution
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph
open SourceForwardGraph (fold_memory_congr)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def fold [Inhabited α] (m : Model α) (stride M N G B group cur : Nat)
    (s : State α) (a : Carries α B) : State α × Carries α B :=
  if G=0 then (s,a) else if cur<M then
    fold m stride M N G B group (cur+G) (storeDX m s stride N B cur)
      (nextCarries m s stride N B group cur a) else (s,a)
termination_by M-cur
decreasing_by omega

theorem range_refines [Inhabited α] (m : Model α) (stride M N G B group cur : Nat)
    (s t : State α) (a : Carries α B) (h : Invariant s N B group a t) :
    ∃ u,FP.Structural.range (engine m) "row" cur M G (SourceGrouped.body stride M N G B) t=some u ∧
      Invariant (fold m stride M N G B group cur s a).1 N B group
        (fold m stride M N G B group cur s a).2 u := by
  rw [FP.Structural.range,fold]
  split
  · exact ⟨t,rfl,h⟩
  · split
    · obtain ⟨u,hu,hnext⟩ := body_refines m s t stride M N G B group cur a h
      simp only [hu,bind,Option.bind]
      exact range_refines m stride M N G B group (cur+G) (storeDX m s stride N B cur) u
        (nextCarries m s stride N B group cur a) hnext
    · exact ⟨t,rfl,h⟩
termination_by M-cur
decreasing_by omega

def stores (s : State α) (N B group : Nat) (a : Carries α B) : State α :=
  let w := (TileShape.allIndices [B]).foldl (fun t i =>
    if i.1.val<N then t.write "DW" (group*N+i.1.val) (.mk .real (a.1 i)) else t) s
  (TileShape.allIndices [B]).foldl (fun t i =>
    if i.1.val<N then t.write "DB" (group*N+i.1.val) (.mk .real (a.2 i)) else t) w

theorem suffix_refines [Inhabited α] (m : Model α) (stride M N G B group : Nat)
    (s t : State α) (a : Carries α B) (h : Invariant s N B group a t) :
    ∃ u,run (engine m) (SourceGrouped.suffix stride M N G B) t=some u ∧
      u.mem=(stores s N B group a).mem := by
  obtain ⟨hm,hu,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb⟩ := h
  simp [SourceGrouped.suffix,SourceGrouped.code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb]
  simp only [stores]
  apply fold_memory_congr
  · apply fold_memory_congr
    · exact hm
    · intro u v i huv
      by_cases hi : i.1.val<N <;> simp [hi,State.write,huv]
  · intro u v i huv
    by_cases hi : i.1.val<N <;> simp [hi,State.write,huv]

def final [Inhabited α] (m : Model α) (stride M N G B : Nat) (s : State α) : State α :=
  let result := fold m stride M N G B (s.pids 0) (s.pids 0) s (zero m B)
  stores result.1 N B (s.pids 0) result.2

theorem source_refines [Inhabited α] (m : Model α) (stride M N G B : Nat) (s : State α) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) s=some t ∧
      t.mem=(final m stride M N G B s).mem := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s stride M N G B
  have hg := hready.2.2.1
  obtain ⟨b,hb,hbready⟩ := range_refines m stride M N G B (s.pids 0) (s.pids 0) s a (zero m B) hready
  obtain ⟨t,ht,hmem⟩ := suffix_refines m stride M N G B (s.pids 0) _ b _ hbready
  refine ⟨t,?_,hmem⟩
  change run (engine m) (SourceGrouped.code stride M N G B) s=some t
  rw [SourceGrouped.code_split,run_append,run_append,ha]
  simpa only [bind,Option.bind,run,step,evalOp_unfold,hg,hb] using ht

#axiomsClean source_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedExecution

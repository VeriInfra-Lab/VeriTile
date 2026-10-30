import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceStrideRefinement
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardStorage

/-! Source launches use the actual observed matrix stride. Complete-memory
refinement permits the independent specifications to use dense logical rows. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceObserved
open VeriTile.Triton FP.Structural SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceStrideRefinement
noncomputable section
open scoped Classical

def runRow [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (row : Nat) : Option (State α) :=
  FP.Structural.exec (engine model)
    (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" view.stride N B eps)
    (SourceForwardStorage.initial api neg c memory row)

def resultRow [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (row : Nat) : State α :=
  (runRow model B eps api neg c memory view row).getD (SourceForwardStorage.initial api neg c memory row)

def values [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) :
    ∀ k : Fin 3,Fin (c.buffers.output k).count → α
  | ⟨0,_⟩,i => ((resultRow model B eps api neg c memory view i.val).mem "Rstd" i.val).read .real
  | ⟨1,_⟩,i => ((resultRow model B eps api neg c memory view i.val).mem "Mean" i.val).read .real
  | ⟨2,_⟩,i => ((resultRow model B eps api neg c memory view (i.val/N)).mem "Y" i.val).read .real

theorem row_success [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N)
    (hn : 0<N) (hb : N≤B) (row : Nat) (hr : row<M) :
    (runRow model B eps api neg c memory view row).isSome=Bool.true ∧
      (resultRow model B eps api neg c memory view row).mem=(SourceForwardStorage.resultRow model B eps api neg c memory row).mem := by
  obtain ⟨t,ht,hm⟩ := SourceStrideRefinement.forward_actual model B eps api neg c memory view hn hb row hr
  obtain ⟨u,hu⟩ := SourceForwardStorage.row_success model B eps api neg c memory hn hb row
  have um := SourceForwardGraph.source_memory model N N B eps hn hb _ u hu
  exact ⟨by simp [runRow,ht],by simpa only [resultRow,runRow,SourceForwardStorage.resultRow,ht,hu,Option.getD_some] using hm.trans um.symm⟩

theorem values_eq [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (hn : 0<N) (hb : N≤B) :
    values model B eps api neg c memory view=SourceForwardStorage.sourceValues model B eps api neg c memory := by
  funext k i
  fin_cases k
  · simp only [values,SourceForwardStorage.sourceValues,(row_success model B eps api neg c memory view hn hb i.val (lt_of_lt_of_eq i.isLt c.inverseCount)).2]
  · simp only [values,SourceForwardStorage.sourceValues,(row_success model B eps api neg c memory view hn hb i.val (lt_of_lt_of_eq i.isLt c.meanCount)).2]
  · have hi : i.val<M*N := lt_of_lt_of_eq i.isLt c.outputCount
    simp only [values,SourceForwardStorage.sourceValues,(row_success model B eps api neg c memory view hn hb (i.val/N) ((Nat.div_lt_iff_lt_mul hn).mpr hi)).2]

def forward [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) : Option (World Store) :=
  if KernelContract.launched M N then
    if ∀ row,row<M → (runRow model B eps api neg c memory view row).isSome then
      some (KernelMemory.run c.buffers api (values model B eps api neg c memory view) memory)
    else none
  else some memory

theorem forward_eq [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (hb : N≤B) :
    forward model B eps api neg c memory view=SourceForwardStorage.exec model B eps api neg c memory := by
  unfold forward SourceForwardStorage.exec
  split
  · rename_i h
    have hn := ((KernelContract.launched_iff M N).mp h).2
    have hs : ∀ row,row<M → (runRow model B eps api neg c memory view row).isSome :=
      fun row hr => (row_success model B eps api neg c memory view hn hb row hr).1
    have hu : ∀ row,row<M → (SourceForwardStorage.runRow model B eps api neg c memory row).isSome := by
      intro row _
      obtain ⟨u,hu⟩ := SourceForwardStorage.row_success model B eps api neg c memory hn hb row
      simp [hu]
    simp only [if_pos hs,if_pos hu,values_eq model B eps api neg c memory view hn hb]
  · rfl

variable {x w dy r mu : Input}

def runGroup [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (group : Nat) : Option (State α) :=
  FP.Structural.exec (engine model)
    (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" view.stride M N (Geometry.groups N) B)
    (SourceBackwardStorage.initialGroup api neg c memory group)

def sourceGroup [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (group : Nat) : State α :=
  (runGroup model B api neg c memory view group).getD (SourceBackwardStorage.initialGroup api neg c memory group)

theorem group_success [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (hn : 0<N) (group : Nat) :
    (runGroup model B api neg c memory view group).isSome=Bool.true ∧
      (sourceGroup model B api neg c memory view group).mem=(SourceBackwardStorage.graphGroup model B api neg c memory group).mem := by
  obtain ⟨t,ht,hm⟩ := SourceGroupedExecution.source_refines model view.stride M N (Geometry.groups N) B
    (SourceBackwardStorage.initialGroup api neg c memory group)
  have hv := hm.trans (SourceStrideRefinement.grouped_memory model view.stride M N (Geometry.groups N) B
    (SourceBackwardStorage.initialGroup api neg c memory group) (view.row_effect hn))
  exact ⟨by simp [runGroup,ht],by simpa only [sourceGroup,runGroup,ht,Option.getD_some] using hv⟩

theorem groupValues_eq [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (hn : 0<N) :
    SourceBackwardStorage.groupValues (sourceGroup model B api neg c memory view) c=
      SourceBackwardStorage.groupValues (SourceBackwardStorage.graphGroup model B api neg c memory) c := by
  funext k i
  fin_cases k <;> simp only [SourceBackwardStorage.groupValues,(group_success model B api neg c memory view hn _).2]

def backward [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) : Option (World Store) :=
  if KernelContract.launched M N then
    if ∀ group,group<min (Geometry.groups N) M → (runGroup model B api neg c memory view group).isSome then
      let after := KernelMemory.run c.first.buffers api (SourceBackwardStorage.groupValues (sourceGroup model B api neg c memory view) c) memory
      if ∀ pid,pid<(N+127)/128 → (SourceBackwardStorage.runFinal model api c after pid).isSome then
        some (KernelMemory.run c.last.buffers api (SourceBackwardStorage.finalValues (SourceBackwardStorage.sourceFinal model api c after) c) after)
      else none
    else none
  else some memory

theorem backward_eq [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) :
    backward model B api neg c memory view=SourceBackwardStorage.exec model B api neg c memory := by
  unfold backward SourceBackwardStorage.exec
  split
  · rename_i h
    have hn := ((KernelContract.launched_iff M N).mp h).2
    have hs : ∀ group,group<min (Geometry.groups N) M → (runGroup model B api neg c memory view group).isSome :=
      fun group _ => (group_success model B api neg c memory view hn group).1
    have hu : ∀ group,group<min (Geometry.groups N) M → (SourceBackwardStorage.runGroup model B api neg c memory group).isSome :=
      fun group _ => (SourceBackwardStorage.group_success model B api neg c memory group).1
    simp only [if_pos hs,if_pos hu,groupValues_eq model B api neg c memory view hn,SourceBackwardStorage.groupValues_eq]
  · rfl

#axiomsClean forward_eq
#axiomsClean backward_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceObserved

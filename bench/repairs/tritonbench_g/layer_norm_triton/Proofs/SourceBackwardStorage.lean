import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedExecution
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalGraph
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelBackward

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardStorage
open VeriTile.Triton FP.Structural SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section
open scoped Classical
variable {x w dy r mu : Input}

def initialGroup [Inhabited α] (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (group : Nat) : State α where
  mem := fun region i => .mk .real (
    if region="X" then inputValue api neg x memory i
    else if region="W" then inputValue api neg w memory i
    else if region="DY" then inputValue api neg dy memory i
    else if region="Mean" then inputValue api neg mu memory i
    else if region="Rstd" then inputValue api neg r memory i
    else if region="DX" then api.read (memory (c.first.buffers.output 0).owner) ((c.first.buffers.output 0).address i)
    else if region="DW" then api.read (memory (c.first.buffers.output 1).owner) ((c.first.buffers.output 1).address i)
    else if region="DB" then api.read (memory (c.first.buffers.output 2).owner) ((c.first.buffers.output 2).address i)
    else default)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then group else 0
  numPids := fun ax => if ax=0 then min (Geometry.groups N) M else 1
  undef := fun dtype _ _ => defaultValue dtype

def runGroup [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (group : Nat) : Option (State α) :=
  FP.Structural.exec (engine model)
    (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" N M N (Geometry.groups N) B)
    (initialGroup api neg c memory group)

def sourceGroup [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (group : Nat) : State α :=
  (runGroup model B api neg c memory group).getD (initialGroup api neg c memory group)

def graphGroup [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (group : Nat) : State α :=
  SourceGroupedExecution.final model N M N (Geometry.groups N) B (initialGroup api neg c memory group)

def groupValues [Inhabited α] (states : Nat → State α)
    (c : KernelContract.Backward M N x w dy r mu) : ∀ k : Fin 3,Fin (c.first.buffers.output k).count → α
  | ⟨0,_⟩,i => ((states ((i.val/N)%Geometry.groups N)).mem "DX" i.val).read .real
  | ⟨1,_⟩,i => ((states (i.val/N)).mem "DW" i.val).read .real
  | ⟨2,_⟩,i => ((states (i.val/N)).mem "DB" i.val).read .real

theorem group_success [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (group : Nat) :
    (runGroup model B api neg c memory group).isSome=Bool.true ∧
      (sourceGroup model B api neg c memory group).mem=(graphGroup model B api neg c memory group).mem := by
  obtain ⟨t,ht,hm⟩ := SourceGroupedExecution.source_refines model N M N (Geometry.groups N) B
    (initialGroup api neg c memory group)
  exact ⟨by simp [runGroup,ht],by simpa only [sourceGroup,runGroup,ht,Option.getD_some] using hm⟩

theorem groupValues_eq [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) :
    groupValues (sourceGroup model B api neg c memory) c=groupValues (graphGroup model B api neg c memory) c := by
  funext k i
  fin_cases k <;> simp only [groupValues,(group_success model B api neg c memory _).2]

def initialFinal [Inhabited α] (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) : State α where
  mem := fun region i => .mk .real (
    if region="DW" then api.read (memory (c.first.buffers.output 1).owner) ((c.first.buffers.output 1).address i)
    else if region="DB" then api.read (memory (c.first.buffers.output 2).owner) ((c.first.buffers.output 2).address i)
    else if region="FINAL_DW" then api.read (memory (c.last.buffers.output 0).owner) ((c.last.buffers.output 0).address i)
    else if region="FINAL_DB" then api.read (memory (c.last.buffers.output 1).owner) ((c.last.buffers.output 1).address i)
    else default)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then pid else 0
  numPids := fun ax => if ax=0 then (N+127)/128 else 1
  undef := fun dtype _ _ => defaultValue dtype

def runFinal [Inhabited α] (model : Model α) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) : Option (State α) :=
  FP.Structural.exec (engine model)
    (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" (min (Geometry.groups N) M) N 32 128)
    (initialFinal api c memory pid)

def sourceFinal [Inhabited α] (model : Model α) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) : State α :=
  (runFinal model api c memory pid).getD (initialFinal api c memory pid)

def graphFinal [Inhabited α] (model : Model α) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) : State α :=
  SourceFinalGraph.final model (initialFinal api c memory pid) (min (Geometry.groups N) M) N 32 128

def finalValues [Inhabited α] (states : Nat → State α)
    (c : KernelContract.Backward M N x w dy r mu) : ∀ k : Fin 2,Fin (c.last.buffers.output k).count → α
  | ⟨0,_⟩,i => ((states (i.val/128)).mem "FINAL_DW" i.val).read .real
  | ⟨1,_⟩,i => ((states (i.val/128)).mem "FINAL_DB" i.val).read .real

theorem final_success [Inhabited α] (model : Model α) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) :
    (runFinal model api c memory pid).isSome=Bool.true ∧
      (sourceFinal model api c memory pid).mem=(graphFinal model api c memory pid).mem := by
  obtain ⟨t,ht,hm⟩ := SourceFinalGraph.source_refines model (initialFinal api c memory pid)
    (min (Geometry.groups N) M) N 32 128
  exact ⟨by simp [runFinal,ht],by simpa only [sourceFinal,runFinal,ht,Option.getD_some] using hm⟩

theorem finalValues_eq [Inhabited α] (model : Model α) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) :
    finalValues (sourceFinal model api c memory) c=finalValues (graphFinal model api c memory) c := by
  funext k i
  fin_cases k <;> simp only [finalValues,(final_success model api c memory _).2]

def afterGroup [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) : World Store :=
  KernelMemory.run c.first.buffers api (groupValues (graphGroup model B api neg c memory) c) memory

def run [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) : World Store :=
  let after := afterGroup model B api neg c memory
  KernelMemory.run c.last.buffers api (finalValues (graphFinal model api c after) c) after

def final [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) : World Store :=
  if KernelContract.launched M N then run model B api neg c memory else memory

/-- Both launches interpret actual typed programs. The second launch reads
the physical scratch memory after the first launch's interpreted stores. -/
def exec [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) : Option (World Store) :=
  if KernelContract.launched M N then
    if ∀ group,group<min (Geometry.groups N) M → (runGroup model B api neg c memory group).isSome then
      let after := KernelMemory.run c.first.buffers api (groupValues (sourceGroup model B api neg c memory) c) memory
      if ∀ pid,pid<(N+127)/128 → (runFinal model api c after pid).isSome then
        some (KernelMemory.run c.last.buffers api (finalValues (sourceFinal model api c after) c) after)
      else none
    else none
  else some memory

theorem exec_eq [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) :
    exec model B api neg c memory=some (final model B api neg c memory) := by
  have hg : ∀ group,group<min (Geometry.groups N) M → (runGroup model B api neg c memory group).isSome :=
    fun group _ => (group_success model B api neg c memory group).1
  have hf (state : World Store) : ∀ pid,pid<(N+127)/128 → (runFinal model api c state pid).isSome :=
    fun pid _ => (final_success model api c state pid).1
  unfold exec final
  split
  · simp only [if_pos hg,groupValues_eq]
    simp only [if_pos (hf _),finalValues_eq,run,afterGroup]
  · rfl

#axiomsClean exec_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardStorage

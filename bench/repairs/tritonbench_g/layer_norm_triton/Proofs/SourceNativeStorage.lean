import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNative
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardStorage

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeStorage
open VeriTile.Triton FP.Structural SourceModel SourceNative SourceNative.Native
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceStrideRefinement
noncomputable section
open scoped Classical

structure Policy where
  reductions : Reductions
  instructions : Instructions
  schedule : OutputSchedule

def Policy.read (p : Policy) := SourceNative.read p.instructions p.schedule

namespace Forward

def initial (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (row : Nat) : State Word :=
  SourceForwardStorage.initial api neg c memory row

def runRow (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (row : Nat) : Option (State Term) :=
  FP.Structural.exec (engine (model p.reductions))
    (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" view.stride N B eps)
    (seed (initial api neg c memory row))

def sourceRow (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (row : Nat) : State Term :=
  (runRow p B eps api neg c memory view row).getD (seed (initial api neg c memory row))

def graphRow (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (row : Nat) : State Term :=
  SourceForwardGraph.final (model p.reductions) N N B eps (seed (initial api neg c memory row))

def values (p : Policy) (states : Nat → State Term) (c : KernelContract.Forward M N) :
    ∀ k : Fin 3,Fin (c.buffers.output k).count → Word
  | ⟨0,_⟩,i => p.read (states i.val) "Rstd" i.val
  | ⟨1,_⟩,i => p.read (states i.val) "Mean" i.val
  | ⟨2,_⟩,i => p.read (states (i.val/N)) "Y" i.val

theorem row_success (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N)
    (hn : 0<N) (hb : N≤B) (row : Nat) (hr : row<M) :
    (runRow p B eps api neg c memory view row).isSome=Bool.true ∧
      (sourceRow p B eps api neg c memory view row).mem=(graphRow p B eps api neg c memory row).mem := by
  obtain ⟨t,ht,hm⟩ := SourceNative.forward p.reductions view.stride N B eps
    (initial api neg c memory row) hn hb (view.row_effect hn row hr)
  exact ⟨by simp [runRow,ht],by simpa only [sourceRow,runRow,ht,Option.getD_some] using hm⟩

theorem values_eq (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N)
    (hn : 0<N) (hb : N≤B) :
    values p (sourceRow p B eps api neg c memory view) c=values p (graphRow p B eps api neg c memory) c := by
  funext k i
  fin_cases k
  · exact read_congr _ _ _ _ (row_success p B eps api neg c memory view hn hb i.val (lt_of_lt_of_eq i.isLt c.inverseCount)).2 _ _
  · exact read_congr _ _ _ _ (row_success p B eps api neg c memory view hn hb i.val (lt_of_lt_of_eq i.isLt c.meanCount)).2 _ _
  · have hi : i.val<M*N := lt_of_lt_of_eq i.isLt c.outputCount
    exact read_congr _ _ _ _ (row_success p B eps api neg c memory view hn hb (i.val/N)
      ((Nat.div_lt_iff_lt_mul hn).mpr hi)).2 _ _

def final (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) : World Store :=
  if KernelContract.launched M N then KernelMemory.run c.buffers api (values p (graphRow p B eps api neg c memory) c) memory else memory

/-- Source expressions are evaluated to native words before physical commit. -/
def exec (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) : Option (World Store) :=
  if KernelContract.launched M N then
    if ∀ row,row<M → (runRow p B eps api neg c memory view row).isSome then
      some (KernelMemory.run c.buffers api (values p (sourceRow p B eps api neg c memory view) c) memory)
    else none
  else some memory

theorem exec_eq (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (hb : N≤B) :
    exec p B eps api neg c memory view=some (final p B eps api neg c memory) := by
  unfold exec final
  split
  · rename_i h
    have hn := ((KernelContract.launched_iff M N).mp h).2
    have hs : ∀ row,row<M → (runRow p B eps api neg c memory view row).isSome :=
      fun row hr => (row_success p B eps api neg c memory view hn hb row hr).1
    simp only [if_pos hs,values_eq p B eps api neg c memory view hn hb]
  · rfl
end Forward

namespace Backward
variable {x w dy r mu : Input}

def runGroup (p : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (group : Nat) : Option (State Term) :=
  FP.Structural.exec (engine (model p.reductions))
    (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" view.stride M N (Geometry.groups N) B)
    (seed (SourceBackwardStorage.initialGroup api neg c memory group))

def sourceGroup (p : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (group : Nat) : State Term :=
  (runGroup p B api neg c memory view group).getD (seed (SourceBackwardStorage.initialGroup api neg c memory group))

def graphGroup (p : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (group : Nat) : State Term :=
  SourceGroupedExecution.final (model p.reductions) N M N (Geometry.groups N) B
    (seed (SourceBackwardStorage.initialGroup api neg c memory group))

def groupValues (p : Policy) (states : Nat → State Term)
    (c : KernelContract.Backward M N x w dy r mu) : ∀ k : Fin 3,Fin (c.first.buffers.output k).count → Word
  | ⟨0,_⟩,i => p.read (states ((i.val/N)%Geometry.groups N)) "DX" i.val
  | ⟨1,_⟩,i => p.read (states (i.val/N)) "DW" i.val
  | ⟨2,_⟩,i => p.read (states (i.val/N)) "DB" i.val

theorem group_success (p : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (hn : 0<N) (group : Nat) :
    (runGroup p B api neg c memory view group).isSome=Bool.true ∧
      (sourceGroup p B api neg c memory view group).mem=(graphGroup p B api neg c memory group).mem := by
  obtain ⟨t,ht,hm⟩ := SourceNative.grouped p.reductions view.stride M N (Geometry.groups N) B
    (SourceBackwardStorage.initialGroup api neg c memory group) (view.row_effect hn)
  exact ⟨by simp [runGroup,ht],by simpa only [sourceGroup,runGroup,ht,Option.getD_some] using hm⟩

theorem groupValues_eq (p : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) (hn : 0<N) :
    groupValues p (sourceGroup p B api neg c memory view) c=groupValues p (graphGroup p B api neg c memory) c := by
  funext k i
  fin_cases k <;> exact read_congr _ _ _ _ (group_success p B api neg c memory view hn _).2 _ _

def runFinal (p : Policy) (api : Memory.API Word Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) : Option (State Term) :=
  FP.Structural.exec (engine (model p.reductions))
    (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" (min (Geometry.groups N) M) N 32 128)
    (seed (SourceBackwardStorage.initialFinal api c memory pid))

def sourceFinal (p : Policy) (api : Memory.API Word Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) : State Term :=
  (runFinal p api c memory pid).getD (seed (SourceBackwardStorage.initialFinal api c memory pid))

def graphFinal (p : Policy) (api : Memory.API Word Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) : State Term :=
  SourceFinalGraph.final (model p.reductions) (seed (SourceBackwardStorage.initialFinal api c memory pid))
    (min (Geometry.groups N) M) N 32 128

def finalValues (p : Policy) (states : Nat → State Term)
    (c : KernelContract.Backward M N x w dy r mu) : ∀ k : Fin 2,Fin (c.last.buffers.output k).count → Word
  | ⟨0,_⟩,i => p.read (states (i.val/128)) "FINAL_DW" i.val
  | ⟨1,_⟩,i => p.read (states (i.val/128)) "FINAL_DB" i.val

theorem final_success (p : Policy) (api : Memory.API Word Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (pid : Nat) :
    (runFinal p api c memory pid).isSome=Bool.true ∧
      (sourceFinal p api c memory pid).mem=(graphFinal p api c memory pid).mem := by
  obtain ⟨t,ht,hm⟩ := SourceNative.final p.reductions (min (Geometry.groups N) M) N
    (SourceBackwardStorage.initialFinal api c memory pid)
  exact ⟨by simp [runFinal,ht],by simpa only [sourceFinal,runFinal,ht,Option.getD_some] using hm⟩

theorem finalValues_eq (p : Policy) (api : Memory.API Word Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) :
    finalValues p (sourceFinal p api c memory) c=finalValues p (graphFinal p api c memory) c := by
  funext k i
  fin_cases k <;> exact read_congr _ _ _ _ (final_success p api c memory _).2 _ _

def afterGroup (p : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) : World Store :=
  KernelMemory.run c.first.buffers api (groupValues p (graphGroup p B api neg c memory) c) memory

def run (p q : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) : World Store :=
  let after := afterGroup p B api neg c memory
  KernelMemory.run c.last.buffers api (finalValues q (graphFinal q api c after) c) after

def final (p q : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) : World Store :=
  if KernelContract.launched M N then run p q B api neg c memory else memory

/-- The final kernel seeds fresh symbolic leaves from the physical words
committed by the grouped kernel, which is an explicit contraction boundary. -/
def exec (p q : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) : Option (World Store) :=
  if KernelContract.launched M N then
    if ∀ group,group<min (Geometry.groups N) M → (runGroup p B api neg c memory view group).isSome then
      let after := KernelMemory.run c.first.buffers api (groupValues p (sourceGroup p B api neg c memory view) c) memory
      if ∀ pid,pid<(N+127)/128 → (runFinal q api c after pid).isSome then
        some (KernelMemory.run c.last.buffers api (finalValues q (sourceFinal q api c after) c) after)
      else none
    else none
  else some memory

theorem exec_eq (p q : Policy) (B : Nat) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) (view : MatrixView x M N) :
    exec p q B api neg c memory view=some (final p q B api neg c memory) := by
  have hf (state : World Store) : ∀ pid,pid<(N+127)/128 → (runFinal q api c state pid).isSome :=
    fun pid _ => (final_success q api c state pid).1
  unfold exec final
  split
  · rename_i h
    have hn := ((KernelContract.launched_iff M N).mp h).2
    have hg : ∀ group,group<min (Geometry.groups N) M → (runGroup p B api neg c memory view group).isSome :=
      fun group _ => (group_success p B api neg c memory view hn group).1
    simp only [if_pos hg,groupValues_eq p B api neg c memory view hn]
    simp only [if_pos (hf _),finalValues_eq,run,afterGroup]
  · rfl
end Backward

#axiomsClean Forward.exec_eq
#axiomsClean Backward.exec_eq
#specNonCircular Forward.graphRow avoiding [FP.Structural.exec]
#specNonCircular Backward.graphGroup avoiding [FP.Structural.exec]
#specNonCircular Backward.graphFinal avoiding [FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeStorage

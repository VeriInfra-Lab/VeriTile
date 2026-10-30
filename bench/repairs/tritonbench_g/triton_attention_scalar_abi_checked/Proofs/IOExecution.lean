import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.IO

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 2000000

theorem mapCell_id {α : Type} (cell : Cell α) : FP.Structural.mapCell id cell=cell := by
  cases cell with
  | mk dtype value=>cases dtype <;> rfl

theorem real_initial (kind : Native.Kind) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (memory : Grid.Memory Erased.R) :
    (realInvocation kind c scale pid).source.initial ((realInvocation kind c scale pid).embedded memory)=
      Grid.entry (Native.gridFor c kind) pid memory := by
  have embedding : (realInvocation kind c scale pid).embed memory=memory := by
    funext r o
    exact mapCell_id (memory r o)
  change Grid.entry (Native.gridFor c kind) pid ((realInvocation kind c scale pid).embed memory)=_
  rw [embedding]

theorem native_initial (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (memory : Grid.Memory Native.Graph) :
    (nativeInvocation one kind c scale owners pid).source.initial
      ((nativeInvocation one kind c scale owners pid).embedded memory)=
      Grid.entry (Native.gridFor c kind) pid (Native.Selected.embed memory) := rfl

theorem real_run (kind : Native.Kind) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (memory : Grid.Memory Erased.R) :
    (realInvocation kind c scale pid).run memory=
      Grid.execute Erased.algebra Erased.model (Native.kernelFor c scale kind) (Native.gridFor c kind) pid memory := by
  unfold MaterializedInvocation.run Invocation.run
  rw [real_initial]
  change (FP.Structural.exec _ _ _).map (fun final=>fun r o=>FP.Structural.mapCell id (final.mem r o))=_
  simp only [mapCell_id]
  rfl

theorem native_run (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (memory : Grid.Memory Native.Graph) :
    (nativeInvocation one kind c scale owners pid).run memory=
      Native.Selected.execute one kind c scale owners pid memory := by
  unfold MaterializedInvocation.run Invocation.run
  rw [native_initial]
  simp only [Native.Selected.execute,Grid.execute,Option.map_map]
  rfl

theorem real_grid (kind : Native.Kind) (c : Shape) (scale : ℝ) (work : List Grid.Program)
    (memory : Grid.Memory Erased.R) :
    runMaterializedSchedule (work.map (realInvocation kind c scale)) memory=
      Grid.run (Grid.execute Erased.algebra Erased.model (Native.kernelFor c scale kind)
        (Native.gridFor c kind)) work memory := by
  induction work generalizing memory with
  | nil=>rfl
  | cons pid work ih=>simp only [List.map_cons,runMaterializedSchedule,Grid.run,real_run,ih]

theorem native_grid (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (work : List Grid.Program) (memory : Grid.Memory Native.Graph) :
    runMaterializedSchedule (work.map (nativeInvocation one kind c scale owners)) memory=
      Grid.run (Native.Selected.execute one kind c scale owners) work memory := by
  induction work generalizing memory with
  | nil=>rfl
  | cons pid work ih=>simp only [List.map_cons,runMaterializedSchedule,Grid.run,native_run,ih]

/-- Equality with the complete previously proved pipeline, not merely a CTA. -/
theorem real_schedule (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (xs : Inputs ℝ c) (memory : Grid.Memory Erased.R) :
    runMaterializedSchedule ((realIO c scale fw pw bw).schedule xs) memory=
      Grid.wholeRun Erased.algebra Erased.model Erased.model Erased.model c scale fw pw bw memory := by
  simp only [realIO,layout,runMaterializedSchedule_append,real_grid,Grid.wholeRun,Native.kernelFor,Native.gridFor]
  simp only [Option.bind_assoc]
  rfl

theorem native_schedule (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (xs : Inputs Native.Word c) (memory : Grid.Memory Native.Graph) :
    runMaterializedSchedule ((nativeIO scalar c owners I literal fw pw bw).schedule xs) memory=
      Native.Selected.run (Native.constexprScale scalar) c (scalar.project scalar.raw) owners fw pw bw memory := by
  simp only [nativeIO,layout,runMaterializedSchedule_append,native_grid,Native.Selected.run]
  simp only [Option.bind_assoc]
  rfl

#axiomsClean real_schedule
#axiomsClean native_schedule
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO

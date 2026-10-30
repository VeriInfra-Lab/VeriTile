import bench.tritonbench_g.rotary_transform_ops.Proofs.SourceSafety
import bench.tritonbench_g.rotary_transform_ops.Proofs.IOGeometry

namespace VeriTile.Bench.Repairs.RotaryTransformOps.IOSafety
open VeriTile.Triton FP.Structural Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem program_safe (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (observe : α → α) (vals : Inputs α c.shape) (bounds : RegionName → Nat)
    (backed : (io p c cu64 off64 observe).InBounds bounds vals)
    (u : State α) (md : ProgramModel.MetadataBinding c u)
    (pid : Program) (inside : pid∈programs c) :
    (job p c.shape cu64 off64 pid).Safe bounds u := by
  have batch : pid.2.1<c.shape.batches :=
    List.mem_range.mp (List.mem_product.mp (List.mem_product.mp inside).2).1
  have head : pid.2.2<c.shape.heads :=
    List.mem_range.mp (List.mem_product.mp (List.mem_product.mp inside).2).2
  let entry := TypedGrid.enter c.shape pid u
  have inputMD := TypedGrid.entered_metadata c u md pid
  have hb : entry.pids 1<c.shape.batches := by simpa [entry,TypedGrid.enter] using batch
  have hh : entry.pids 2<c.shape.heads := by simpa [entry,TypedGrid.enter] using head
  change TraceSafe (KernelAlgebra.backend p) bounds (TypedExecution.kernel c.shape cu64 off64) entry
  apply SourceSafety.source_safe
  · intro packed
    exact backed.1.2.1 (3 : Fin 6) (⟨entry.pids 1+1,by omega⟩,()) packed
  · intro tensor
    exact backed.1.2.1 (4 : Fin 6) (⟨entry.pids 1,hb⟩,()) tensor
  · exact IOGeometry.data_input p c valid cu64 off64 observe vals bounds backed entry inputMD hb hh
  · exact IOGeometry.data_output p c valid cu64 off64 observe vals bounds backed entry inputMD hb hh
  · intro lane pair active
    exact backed.1.2.1 (1 : Fin 6)
      (⟨_,IOGeometry.coefficient c entry inputMD hb lane pair active⟩,()) trivial
  · intro lane pair active
    exact backed.1.2.1 (2 : Fin 6)
      (⟨_,IOGeometry.coefficient c entry inputMD hb lane pair active⟩,()) trivial

theorem list_safe (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (observe : α → α) (vals : Inputs α c.shape) (bounds : RegionName → Nat)
    (backed : (io p c cu64 off64 observe).InBounds bounds vals)
    (ps : List Program) (inside : ∀ pid∈ps,pid∈programs c)
    (u : State α) (md : ProgramModel.MetadataBinding c u) :
    ScheduleSafe bounds (ps.map (job p c.shape cu64 off64)) u := by
  induction ps generalizing u with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨program_safe p c valid cu64 off64 observe vals bounds backed u md pid
      (inside pid (by simp)),?_⟩
    intro next ran
    let entry := TypedGrid.enter c.shape pid u
    have inputMD := TypedGrid.entered_metadata c u md pid
    change exec (KernelAlgebra.backend p) (TypedExecution.kernel c.shape cu64 off64) entry=some next at ran
    have mem : next.mem=(ProgramModel.result c.shape p entry).mem := by
      have got := TypedExecution.memory_result p c.shape cu64 off64 entry
      simpa only [ran,Option.map_some,Option.some.injEq] using got
    exact ih (fun q hq => inside q (by simp [hq])) next
      (TypedGrid.metadata_mem_eq c next _ mem (TypedGrid.metadata_after_result c p entry inputMD))

theorem whole (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (observe : α → α) (vals : Inputs α c.shape) (ready : Ready c vals)
    (bounds : RegionName → Nat) (backed : (io p c cu64 off64 observe).InBounds bounds vals)
    (u : State α) (hin : (io p c cu64 off64 observe).InputBinding vals u) :
    ScheduleSafe bounds (Interface.schedule p c cu64 off64) u := by
  unfold Interface.schedule
  split_ifs
  · trivial
  · exact list_safe p c valid cu64 off64 observe vals bounds backed (programs c)
      (fun _ h => h) u (IOFacts.metadata p c cu64 off64 observe vals ready u hin)

end
end VeriTile.Bench.Repairs.RotaryTransformOps.IOSafety

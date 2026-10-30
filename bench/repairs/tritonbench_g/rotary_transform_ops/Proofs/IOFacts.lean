import bench.repairs.tritonbench_g.rotary_transform_ops.IO
import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.KernelEndpoints
import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Bounds

namespace VeriTile.Bench.Repairs.RotaryTransformOps.IOFacts
open VeriTile.Triton FP.Structural Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem job_initial (p : Ops α) (s : Shape) (cu64 off64 : Bool)
    (pid : Program) (u : State α) :
    (job p s cu64 off64 pid).initial u=TypedGrid.enter s pid u := rfl

theorem execute_eq (p : Ops α) (s : Shape) (cu64 off64 : Bool)
    (ps : List Program) (u : State α) :
    runSchedule (ps.map (job p s cu64 off64)) u=TypedGrid.execute p s cu64 off64 ps u := by
  induction ps generalizing u with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,TypedGrid.execute,Invocation.run,job_initial,job]
    congr 1
    funext next
    exact ih next

theorem schedule_eq (p : Ops α) (c : Config) (cu64 off64 : Bool) (u : State α) :
    runSchedule (Interface.schedule p c cu64 off64) u=TypedGrid.run p c cu64 off64 u := by
  unfold Interface.schedule TypedGrid.run
  split_ifs
  · rfl
  · exact execute_eq p c.shape cu64 off64 _ u

theorem metadata (p : Ops α) (c : Config) (cu64 off64 : Bool) (observe : α → α)
    (vals : Inputs α c.shape) (ready : Ready c vals) (u : State α)
    (hin : (io p c cu64 off64 observe).InputBinding vals u) :
    ProgramModel.MetadataBinding c u := by
  constructor
  · intro hv b hb
    exact (hin (3 : Fin 6) (⟨b,by omega⟩,()) hv).trans (ready.1 hv _)
  · intro ht b hb
    exact (hin (4 : Fin 6) (⟨b,hb⟩,()) ht).trans (ready.2.1 ht _)

theorem suffix (p : Ops α) (c : Config) (cu64 off64 : Bool) (observe : α → α)
    (vals : Inputs α c.shape) (ready : Ready c vals) (u : State α)
    (hin : (io p c cu64 off64 observe).InputBinding vals u) :
    KernelEndpoints.TailReady c u := by
  intro token head col hn hh hd tail
  let i : Fin (Host.size c.shape) := ⟨_,Host.address_bound _ token head col hn hh hd⟩
  have mask : 2*c.shape.half ≤ i.val%c.shape.width := by
    simpa only [i,Host.address_mod c.shape token head col hd] using tail
  exact (hin (5 : Fin 6) (i,()) mask).trans
    ((ready.2.2 i mask).trans (hin (0 : Fin 6) (i,()) trivial).symm)

theorem input_value (p : Ops α) (c : Config) (cu64 off64 : Bool) (observe : α → α)
    (vals : Inputs α c.shape) (u : State α)
    (hin : (io p c cu64 off64 observe).InputBinding vals u)
    (offset : Nat) (inside : offset<Host.size c.shape) :
    KernelAlgebra.memory u .input offset=Interface.inputMemory vals .input offset := by
  simpa only [Interface.inputMemory,dif_pos inside] using
    hin (0 : Fin 6) (⟨offset,inside⟩,()) trivial

theorem cosine_value (p : Ops α) (c : Config) (cu64 off64 : Bool) (observe : α → α)
    (vals : Inputs α c.shape) (u : State α)
    (hin : (io p c cu64 off64 observe).InputBinding vals u)
    (offset : Nat) (inside : offset<c.shape.coefficientRows*c.shape.half) :
    KernelAlgebra.memory u .cosine offset=Interface.inputMemory vals .cosine offset := by
  simpa only [Interface.inputMemory,dif_pos inside] using
    hin (1 : Fin 6) (⟨offset,inside⟩,()) trivial

theorem sine_value (p : Ops α) (c : Config) (cu64 off64 : Bool) (observe : α → α)
    (vals : Inputs α c.shape) (u : State α)
    (hin : (io p c cu64 off64 observe).InputBinding vals u)
    (offset : Nat) (inside : offset<c.shape.coefficientRows*c.shape.half) :
    KernelAlgebra.memory u .sine offset=Interface.inputMemory vals .sine offset := by
  simpa only [Interface.inputMemory,dif_pos inside] using
    hin (2 : Fin 6) (⟨offset,inside⟩,()) trivial

theorem outside (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (observe : α → α) (vals : Inputs α c.shape) (r : RegionName) (o : Nat)
    (h : (io p c cu64 off64 observe).toProgramIO.Outside vals r o) :
    TypedGrid.Outside c r o := by
  intro w hw written
  rcases written.2 with low | high
  all_goals
    have hd := dim_bound c.shape w.pair hw.2.2.2
    let lane (second : Bool) : TileIndex [c.shape.batches,c.shape.maximum,c.shape.heads,c.shape.width] :=
      (⟨w.batch,hw.1⟩,⟨w.row,lt_of_lt_of_le hw.2.1 (valid.lengthBound _ hw.1)⟩,
        ⟨w.head,hw.2.2.1⟩,⟨c.shape.dim w.pair second,lt_of_lt_of_le (hd second) valid.rotary⟩,())
  · have contradiction := h.1 (0 : Fin 1) (lane Bool.false) ⟨hw.2.1,hd Bool.false⟩
    exact contradiction.elim (fun h => h written.1) (fun h => h low)
  · have contradiction := h.1 (0 : Fin 1) (lane Bool.true) ⟨hw.2.1,hd Bool.true⟩
    exact contradiction.elim (fun h => h written.1) (fun h => h high)

end
end VeriTile.Bench.Repairs.RotaryTransformOps.IOFacts

import bench.repairs.tritonbench_g.rotary_transform_tile_checked.IO
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.KernelEndpoints
import VeriTile.Triton.Float.ReadProgramIO

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem initial_eq (p : Ops α) (c : Config) (cu64 off64 : Bool) (pid : Program) (s : State α) :
    (invocation p c cu64 off64 pid).initial s=TypedGrid.enter c.shape pid s := rfl

theorem list_eq (p : Ops α) (c : Config) (cu64 off64 : Bool) (jobs : List Program) (s : State α) :
    runSchedule (jobs.map (invocation p c cu64 off64)) s=
      TypedGrid.execute p c.shape cu64 off64 jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons,runSchedule,TypedGrid.execute]
    change (exec (KernelAlgebra.backend p) (TypedExecution.kernel c.shape cu64 off64)
      (TypedGrid.enter c.shape pid s)).bind _ = _
    cases h : exec (KernelAlgebra.backend p) (TypedExecution.kernel c.shape cu64 off64)
      (TypedGrid.enter c.shape pid s) <;> simp [ih]

theorem schedule_eq (p : Ops α) (obs : α → α) (c : Config) (cu64 off64 : Bool)
    (vals : ChannelValues α (inputs c)) (s : State α) :
    runSchedule ((io p obs c cu64 off64).schedule vals) s=TypedGrid.run p c cu64 off64 s := by
  by_cases empty : c.shape.total=0 ∨ c.shape.half=0
  · simp [io,empty,runSchedule,TypedGrid.run]
  · simpa only [io,if_neg empty,TypedGrid.run] using list_eq p c cu64 off64 (programs c) s

theorem metadata_binding (p : Ops α) (obs : α → α) (c : Config) (cu64 off64 : Bool)
    (vals : ChannelValues α (inputs c)) (legal : domain c vals) (initial : State α)
    (hin : (io p obs c cu64 off64).InputBinding vals initial) :
    ProgramModel.MetadataBinding c initial := by
  constructor
  · intro hv i hi
    have h := hin ⟨3,by simp [io,inputs]⟩ (⟨i,by omega⟩,()) hv
    exact h.trans (legal.1 hv ⟨i,by omega⟩)
  · intro ht i hi
    have h := hin ⟨4,by simp [io,inputs]⟩ (⟨i,hi⟩,()) ht
    exact h.trans (legal.2.1 ht ⟨i,hi⟩)

theorem tail_ready (p : Ops α) (obs : α → α) (c : Config) (cu64 off64 : Bool)
    (vals : ChannelValues α (inputs c)) (legal : domain c vals) (initial : State α)
    (hin : (io p obs c cu64 off64).InputBinding vals initial) :
    KernelEndpoints.TailReady c initial := by
  intro token head col ht hh hc tail
  have bound : c.shape.address token head col<dataSize c.shape := Host.address_bound c.shape _ _ _ ht hh hc
  have mod : c.shape.address token head col%c.shape.width=col := by
    simp [Shape.address,Nat.add_mod,Nat.mod_eq_of_lt hc]
  have mask : 2*c.shape.half ≤ c.shape.address token head col%c.shape.width := by simpa [mod]
  have hout := hin ⟨5,by simp [io,inputs]⟩ (⟨_,bound⟩,()) mask
  have hx := hin ⟨0,by simp [io,inputs]⟩ (⟨_,bound⟩,()) trivial
  have eq := hout.trans ((legal.2.2 ⟨_,bound⟩ mask).trans hx.symm)
  cases inplace : c.shape.inplace <;>
    simpa [io,inputs,TypedExecution.outputRegion,KernelAlgebra.regionName,Shape.output,inplace] using eq

theorem output_region (s : Shape) :
    TypedExecution.outputRegion s=KernelAlgebra.regionName s.output := by
  cases hi : s.inplace <;> simp [TypedExecution.outputRegion,KernelAlgebra.regionName,Shape.output,hi]

/-- The declared write window is exactly the original complete-grid frame. -/
theorem frame (p : Ops α) (obs : α → α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (vals : ChannelValues α (inputs c)) (initial final : State α)
    (unchanged : ∀ r off,TypedGrid.Outside c r off → final.mem r off=initial.mem r off) :
    (io p obs c cu64 off64).Frame vals initial final := by
  intro r off outside
  apply unchanged
  intro w hw written
  have row : w.row<c.shape.maximum := lt_of_lt_of_le hw.2.1 (valid.lengthBound w.batch hw.1)
  have miss (second : Bool) : ¬(r=KernelAlgebra.regionName c.shape.output ∧ off=c.cell w second) := by
    have col := dim_bound c.shape w.pair hw.2.2.2 second
    have width := lt_of_lt_of_le col valid.rotary
    have h := outside.1 ⟨0,by simp [io,outputs]⟩
      (⟨w.batch,hw.1⟩,⟨w.row,row⟩,⟨w.head,hw.2.2.1⟩,⟨c.shape.dim w.pair second,width⟩,()) ⟨hw.2.1,col⟩
    simpa only [io,outputs,output_region,Config.cell,not_and_or,Ne] using h
  rcases written.2 with low|high
  · exact miss Bool.false ⟨written.1,low⟩
  · exact miss Bool.true ⟨written.1,high⟩

#axiomsClean metadata_binding
#axiomsClean tail_ready
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO

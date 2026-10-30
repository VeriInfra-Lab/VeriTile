import bench.tritonbench_g.rotary_transform.Proofs.CompleteSafety
import bench.tritonbench_g.rotary_transform.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem access_bounds (c : Config) (valid : c.Valid) (bounds : RegionName → Nat)
    (initial : State α) (md : ProgramModel.MetadataBinding c initial)
    (batch : initial.pids 1<c.shape.batches) (head : initial.pids 2<c.shape.heads)
    (hx : ∀ i,i<dataSize c.shape → i<bounds "x")
    (hco : ∀ i,i<coefficientSize c.shape → i<bounds "cos")
    (hsn : ∀ i,i<coefficientSize c.shape → i<bounds "sin")
    (ho : ∀ b row h col,b<c.shape.batches → row<c.length b → h<c.shape.heads → col<2*c.shape.half →
      c.shape.address (c.start b+row) h col<bounds (TypedExecution.outputRegion c.shape)) :
    SourceSafety.AccessBounds c.shape initial bounds := by
  constructor
  · intro lane col live
    have row : ProgramModel.row c.shape initial lane<c.length (initial.pids 1) := by
      simpa only [ProgramModel.length_eq c valid initial md batch,Int.ofNat_lt] using live.1
    have colBound : col<c.shape.width := lt_of_lt_of_le live.2 valid.rotary
    rw [ProgramModel.offset_eq c initial md batch]
    exact ⟨hx _ (Host.address_bound c.shape _ _ _
      (row_bound c valid _ _ batch row) head colBound),ho _ _ _ _ batch row head live.2⟩
  · intro lane pair live
    have row : ProgramModel.row c.shape initial lane+c.shift (initial.pids 1)<c.shape.coefficientRows := by
      have := live.1
      simpa only [ProgramModel.shift_eq c initial md batch,←Int.natCast_add,Int.ofNat_lt] using this
    rw [ProgramModel.coefficient_eq c initial md batch]
    have bound : c.coeff (ProgramModel.work c.shape initial lane pair)<coefficientSize c.shape := by
      change (ProgramModel.row c.shape initial lane+c.shift (initial.pids 1))*c.shape.half+pair<
        c.shape.coefficientRows*c.shape.half
      calc
        _ < (ProgramModel.row c.shape initial lane+c.shift (initial.pids 1))*c.shape.half+c.shape.half :=
          Nat.add_lt_add_left live.2 _
        _ = (ProgramModel.row c.shape initial lane+c.shift (initial.pids 1)+1)*c.shape.half := by ring
        _ ≤ _ := Nat.mul_le_mul_right _ row
    exact ⟨hco _ bound,hsn _ bound⟩

theorem list_safe (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (bounds : RegionName → Nat)
    (hx : ∀ i,i<dataSize c.shape → i<bounds "x")
    (hco : ∀ i,i<coefficientSize c.shape → i<bounds "cos")
    (hsn : ∀ i,i<coefficientSize c.shape → i<bounds "sin")
    (hcu : c.shape.varlen=Bool.true → ∀ i,i≤c.shape.batches → i<bounds "cu")
    (hoff : c.shape.offsetTensor=Bool.true → ∀ i,i<c.shape.batches → i<bounds "offsets")
    (ho : ∀ b row h col,b<c.shape.batches → row<c.length b → h<c.shape.heads → col<2*c.shape.half →
      c.shape.address (c.start b+row) h col<bounds (TypedExecution.outputRegion c.shape))
    (jobs : List Program) (legal : ∀ pid∈jobs,pid∈programs c)
    (initial : State α) (md : ProgramModel.MetadataBinding c initial) :
    ScheduleSafe bounds (jobs.map (invocation p c cu64 off64)) initial := by
  induction jobs generalizing initial with
  | nil => trivial
  | cons pid jobs ih =>
    have member := legal pid (by simp)
    have batch : pid.2.1<c.shape.batches :=
      List.mem_range.mp (List.mem_product.mp (List.mem_product.mp member).2).1
    have head : pid.2.2<c.shape.heads :=
      List.mem_range.mp (List.mem_product.mp (List.mem_product.mp member).2).2
    let entry := TypedGrid.enter c.shape pid initial
    have b : entry.pids 1<c.shape.batches := by simpa [entry,TypedGrid.enter] using batch
    have h : entry.pids 2<c.shape.heads := by simpa [entry,TypedGrid.enter] using head
    have emd : ProgramModel.MetadataBinding c entry := TypedGrid.entered_metadata c initial md pid
    refine ⟨SourceSafety.source_safe (KernelAlgebra.backend p) c.shape bounds entry
      (access_bounds c valid bounds entry emd b h hx hco hsn ho)
      (fun hv => hcu hv _ (by omega)) (fun ht => hoff ht _ b) cu64 off64,?_⟩
    intro next ran
    have mem : next.mem=(ProgramModel.result c.shape p entry).mem := by
      have got := TypedExecution.memory_result p c.shape cu64 off64 entry
      change exec (KernelAlgebra.backend p) (TypedExecution.kernel c.shape cu64 off64) entry=some next at ran
      simpa only [ran,Option.map_some,Option.some.injEq] using got
    exact ih (fun q hq => legal q (by simp [hq])) next
      (TypedGrid.metadata_mem_eq c next _ mem (TypedGrid.metadata_after_result c p entry emd))

theorem schedule_safe (p : Ops α) (obs : α → α) (c : Config) (valid : c.Valid)
    (cu64 off64 : Bool) (vals : ChannelValues α (inputs c)) (legal : domain c vals)
    (bounds : RegionName → Nat) (bound : (io p obs c cu64 off64).InBounds bounds vals)
    (initial : State α) (hin : (io p obs c cu64 off64).InputBinding vals initial) :
    ScheduleSafe bounds ((io p obs c cu64 off64).schedule vals) initial := by
  by_cases empty : c.shape.total=0 ∨ c.shape.half=0
  · simp [io,empty,ScheduleSafe]
  · simp only [io,if_neg empty]
    apply list_safe p c valid cu64 off64 bounds
    · intro i hi
      exact bound.1.2.1 ⟨0,by simp [io,inputs]⟩ (⟨i,hi⟩,()) trivial
    · intro i hi
      exact bound.1.2.1 ⟨1,by simp [io,inputs]⟩ (⟨i,hi⟩,()) trivial
    · intro i hi
      exact bound.1.2.1 ⟨2,by simp [io,inputs]⟩ (⟨i,hi⟩,()) trivial
    · intro hv i hi
      exact bound.1.2.1 ⟨3,by simp [io,inputs]⟩ (⟨i,by omega⟩,()) hv
    · intro ht i hi
      exact bound.1.2.1 ⟨4,by simp [io,inputs]⟩ (⟨i,hi⟩,()) ht
    · intro b row h col hb hr hh hc
      have rowMax : row<c.shape.maximum := lt_of_lt_of_le hr (valid.lengthBound b hb)
      have colWidth : col<c.shape.width := lt_of_lt_of_le hc valid.rotary
      exact bound.1.2.2.1 ⟨0,by simp [io,outputs]⟩
        (⟨b,hb⟩,⟨row,rowMax⟩,⟨h,hh⟩,⟨col,colWidth⟩,()) ⟨hr,hc⟩
    · exact fun _ h => h
    · exact metadata_binding p obs c cu64 off64 vals legal initial hin

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO

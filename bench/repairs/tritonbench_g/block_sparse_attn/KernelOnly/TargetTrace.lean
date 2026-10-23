import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.GridExecution

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API Apart Write)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
set_option linter.unusedSectionVars Bool.false
namespace PhysicalKernel
variable {α : Type} [Inhabited α]

def activeLanes (p : Parameters) (id : Program) : List (TileIndex [16,16]) :=
  (TileShape.allIndices [16,16]).filter (fun i => decide (id.row*16+i.1.val<p.length))

theorem activeLanes_mem (p : Parameters) (id : Program) (i : TileIndex [16,16]) :
    i∈activeLanes p id ↔ id.row*16+i.1.val<p.length := by
  constructor
  · intro member
    exact of_decide_eq_true (List.mem_filter.mp member).2
  · intro live
    exact List.mem_filter.mpr ⟨TileShape.mem_allIndices _ _,decide_eq_true live⟩

def targetEvents (p : Parameters) (batches : Nat) (buffers : Physical.Buffers)
    (f : Work → α) (id : Program) : List (Write α) :=
  (activeLanes p id).map (fun i =>
    ⟨Physical.outputAddress buffers (shape p batches) (coordinate p batches id i),
      f (coordinate p batches id i)⟩)

def targetTrace (p : Parameters) (batches : Nat) (buffers : Physical.Buffers)
    (f : Work → α) : List (Write α) := (programs p batches).flatMap (targetEvents p batches buffers f)

theorem targetTrace_mem (p : Parameters) (batches : Nat) (buffers : Physical.Buffers)
    (f : Work → α) (heads : 0<p.heads) (e : Write α) :
    e∈targetTrace p batches buffers f ↔ ∃ w, w.active (shape p batches) ∧
      e=⟨Physical.outputAddress buffers (shape p batches) w,f w⟩ := by
  constructor
  · intro member
    obtain ⟨id,hid,eMember⟩ := List.mem_flatMap.mp member
    obtain ⟨i,hi,rfl⟩ := List.mem_map.mp eMember
    exact ⟨coordinate p batches id i,coordinate_active p batches id
      ((programs_mem p batches id).mp hid) heads i ((activeLanes_mem p id i).mp hi),rfl⟩
  · rintro ⟨w,live,rfl⟩
    obtain ⟨id,hid,i,hi,equal⟩ := coordinate_coverage p batches w live
    apply List.mem_flatMap.mpr
    refine ⟨id,hid,List.mem_map.mpr ⟨i,(activeLanes_mem p id i).mpr hi,?_⟩⟩
    rw [equal]

/-- Internal bridge from already-proved per-lane mathematics to physical
events. Its pointwise premise is discharged by `output_view`/`output_real`. -/
theorem gridEvents_target (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) (f : Work → α)
    (values : ∀ id, id.Valid p batches → ∀ i : TileIndex [16,16],id.row*16+i.1.val<p.length →
      output a native p (state access a.zero buffers initial id) i (owner i.2.1.val)=
        f (coordinate p batches id i)) :
    gridEvents access a native p buffers initial (programs p batches)=targetTrace p batches buffers f := by
  unfold gridEvents targetTrace
  apply List.flatMap_congr
  intro id member
  change List.map _ (activeLanes p id)=List.map _ (activeLanes p id)
  apply List.map_congr_left
  intro i hi
  have live := (activeLanes_mem p id i).mp hi
  have value := values id ((programs_mem p batches id).mp member) i live
  simp only [output_offset access a.zero buffers initial p batches id i abi.parameters,value]
  rfl

theorem target_readback (api : API α Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (f : Work → α) (heads : 0<p.heads) (initial : Store)
    (w : Work) (live : w.active (shape p batches)) :
    api.read (LayernormFwdTriton.Memory.run api (targetTrace p batches buffers f) initial)
      (Physical.outputAddress buffers (shape p batches) w)=f w := by
  apply LayernormFwdTriton.Memory.run_readback
  · exact ⟨_,(targetTrace_mem p batches buffers f heads _).mpr ⟨w,live,rfl⟩,rfl,rfl⟩
  · intro e member
    obtain ⟨u,hu,rfl⟩ := (targetTrace_mem p batches buffers f heads e).mp member
    by_cases equal : u=w
    · subst u; exact Or.inl ⟨rfl,rfl⟩
    · apply Or.inr
      have different : u.target (shape p batches)≠w.target (shape p batches) :=
        fun same => equal (target_injective (shape p batches) u w hu live same)
      unfold Apart Physical.outputAddress Physical.inputAddress
      dsimp only
      omega

def Outside (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (address : Nat) : Prop :=
  ∀ w,w.active (shape p batches) → address<Physical.outputAddress buffers (shape p batches) w ∨
    Physical.outputAddress buffers (shape p batches) w+4≤address

theorem target_frame (api : API α Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (f : Work → α) (heads : 0<p.heads) (initial : Store)
    (address : Nat) (outside : Outside p batches buffers address) :
    api.byte (LayernormFwdTriton.Memory.run api (targetTrace p batches buffers f) initial) address=
      api.byte initial address := by
  apply LayernormFwdTriton.Memory.run_byte_frame
  intro e member
  obtain ⟨w,hw,rfl⟩ := (targetTrace_mem p batches buffers f heads e).mp member
  exact outside w hw

end PhysicalKernel
#axiomsClean PhysicalKernel.activeLanes_mem
#axiomsClean PhysicalKernel.targetTrace_mem
#axiomsClean PhysicalKernel.gridEvents_target
#axiomsClean PhysicalKernel.target_readback
#axiomsClean PhysicalKernel.target_frame
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

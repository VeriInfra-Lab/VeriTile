/- Historical padded/snapshot wrapper theorem, retained as prior proof credit. -/
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.FPScalarTree
import bench.repairs.tritonbench_g.bgmv_expand_slice.Kernels
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.FPHost
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Preparation
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.HostProtocol
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.NativeGrid

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.FPEquiv
open VeriTile.Triton FP.Structural FPValues Snapshot ReturnedStorage
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

abbrev caller (codec : LogicalReturn.Codec α) (views : CallerViews.Views) (memory : CallerViews.Memory α) :=
  CallerViews.logical codec views memory

def scratchRead (h : Call) (storage : Storage.Placement h) (heap : Nat → α) (batch row : Nat) : α :=
  heap (storage.output.address (batch*(h.columns+255)+row+h.slice))

/-- Complete FP32/FP16/FP32 source execution, actual native scratch readback,
and every changed byte of the same supplied output object. The numerical
reference contains the unchanged casts/multiply/reduction/add order. No Real
execution theorem, successful-execution premise or whole-output oracle occurs. -/
theorem complete (M : Algebra α) (reduction : Reduction α) (h : Call) (primary : Kernels.Primary h)
    (schedule : Schedule h) (abi : NativeGrid.ABI h schedule) (codec : LogicalReturn.Codec α) (views : CallerViews.Views)
    (memory : CallerViews.Memory α) (zero : α)
    (packed : Packed α h (caller codec views memory) zero)
    (other : RegionName → Nat → Cell α)
    (padding : (dtype : TileDType) → RegionName → Nat → Value α dtype)
    (ready : Preparation.Prepared α h (FPHost.data h (caller codec views memory) zero packed other padding))
    (allocations : CallerViews.Allocations views) (viewFacts : CallerViews.Valid h views allocations)
    (fresh : Preparation.Fresh ready (CallerViews.Owned views allocations))
    (guard : ViewGuard.Admitted views.output) (encode : Encode α) (before : Bytes)
    (initialOutput : CallerViews.InitialOutput codec encode views memory before)
    (valid : ∀ b,b<h.batches → (caller codec views memory).indices b=-1 ∨
      (0 ≤ (caller codec views memory).indices b ∧ ((caller codec views memory).indices b).toNat<h.adapters)) :
    let initial := FPHost.data h (caller codec views memory) zero packed other padding
    let native := PhysicalGrid.run M reduction h ready.storage schedule.programs initial (Preparation.outputHeap ready)
    let returned := ReturnedStorage.returned encode h views.output (caller codec views memory).indices
      (scratchRead h ready.storage native) before
    FPGrid.run M reduction h schedule.programs initial=some (FPGrid.fold M reduction h schedule.programs initial) ∧
    returned.result=() ∧ returned.outputObject=views.output.object ∧ returned.outputOwner=views.output.owner ∧
    (∀ p : Geometry.Program h,∀ n : Fin 256,
      (NativeGrid.outputPointer h ready.storage schedule abi p n).toNat=
        ready.storage.output.address (Geometry.outputOffset h p.batch p.row n.val)) ∧
    (∀ address,CallerViews.Owned views allocations address → native address=Preparation.outputHeap ready address) ∧
    (∀ i : Coord views.output,∀ byte : Fin 4,views.output.address i byte<2^64) ∧
    (∀ i : Coord views.output,Active h views.output (caller codec views memory).indices i → ∀ byte : Fin 4,
      returned.memory views.output.owner (views.output.address i byte)=
        encode views.output.negative
          (FPHost.reference M reduction h (caller codec views memory) zero i.1.val
            ((caller codec views memory).indices i.1.val).toNat (i.2.val-h.slice)) byte) ∧
    (∀ i : Coord views.output,¬Active h views.output (caller codec views memory).indices i → ∀ byte : Fin 4,
      returned.memory views.output.owner (views.output.address i byte)=
        encode views.output.negative ((caller codec views memory).output i.1.val i.2.val) byte) ∧
    (∀ owner address,(¬∃ i : Coord views.output,∃ byte : Fin 4,
        Active h views.output (caller codec views memory).indices i ∧ owner=views.output.owner ∧
        address=views.output.address i byte) → returned.memory owner address=before owner address) := by
  dsimp only
  have validity : ∀ b,b<h.batches →
      (FPHost.data h (caller codec views memory) zero packed other padding).indices b=-1 ∨
      0 ≤ (FPHost.data h (caller codec views memory) zero packed other padding).indices b := by
    intro b hb
    change packed.indices b=-1 ∨ 0 ≤ packed.indices b
    rw [packed.metadataCopy b hb]
    exact (valid b hb).imp_right And.left
  refine ⟨FPGrid.executes M reduction h primary schedule.programs _ validity,rfl,rfl,rfl,?_,?_,?_,?_,?_,?_⟩
  · exact NativeGrid.output_readback h ready.storage schedule abi
  · intro address owned
    exact Preparation.caller_frame M reduction ready _ fresh schedule.programs address owned
  · intro i byte
    exact (ReturnedStorage.backed_byte views.output viewFacts.output i byte).2.2
  · intro i active byte
    dsimp only [ReturnedStorage.returned]
    rw [ReturnedStorage.copy_byte encode h views.output (ViewGuard.writable views.output guard)
      (caller codec views memory).indices _ before i active byte]
    let p := HostCoordinates.program h views.output (caller codec views memory).indices i active
    have activePacked : packed.indices p.batch≠-1 := by
      rw [packed.metadataCopy p.batch p.batchInside]
      exact active.2.2.2
    have physical := PhysicalGrid.value M reduction h ready.storage schedule
      (FPHost.data h (caller codec views memory) zero packed other padding) (Preparation.outputHeap ready)
      (Preparation.output_bound ready) p activePacked
    have selected : ((caller codec views memory).indices p.batch).toNat<h.adapters :=
      ((valid p.batch p.batchInside).resolve_left active.2.2.2).2
    rw [FPHost.value_snapshot M reduction h _ zero packed other padding p selected] at physical
    exact congrArg (fun value => encode views.output.negative value byte) physical
  · intro i inactive byte
    dsimp only [ReturnedStorage.returned]
    rw [ReturnedStorage.inactive_byte encode h views.output (ViewGuard.writable views.output guard)
      (caller codec views memory).indices _ before i inactive byte]
    exact initialOutput i byte
  · intro owner address outside
    exact ReturnedStorage.copy_frame encode h views.output _ _ before owner address outside

/-- Whole source/grid/returned-byte result for the explicit typed scalar-tree
profile. Native use requires only the listed local scalar correspondences and
structural matching of the chosen compiler schedule; there is no numerical
assumption about a whole reduction result. All original caller domains remain. -/
theorem complete_tree (M : Algebra α) (plans : FPScalarTree.Plans) (h : Call) (primary : Kernels.Primary h)
    (schedule : Schedule h) (abi : NativeGrid.ABI h schedule) (codec : LogicalReturn.Codec α) (views : CallerViews.Views)
    (memory : CallerViews.Memory α) (zero : α)
    (packed : Packed α h (caller codec views memory) zero)
    (other : RegionName → Nat → Cell α)
    (padding : (dtype : TileDType) → RegionName → Nat → Value α dtype)
    (ready : Preparation.Prepared α h (FPHost.data h (caller codec views memory) zero packed other padding))
    (allocations : CallerViews.Allocations views) (viewFacts : CallerViews.Valid h views allocations)
    (fresh : Preparation.Fresh ready (CallerViews.Owned views allocations))
    (guard : ViewGuard.Admitted views.output) (encode : Encode α) (before : Bytes)
    (initialOutput : CallerViews.InitialOutput codec encode views memory before)
    (valid : ∀ b,b<h.batches → (caller codec views memory).indices b=-1 ∨
      (0 ≤ (caller codec views memory).indices b ∧ ((caller codec views memory).indices b).toNat<h.adapters)) :
    let initial := FPHost.data h (caller codec views memory) zero packed other padding
    let native := PhysicalGrid.run M (FPScalarTree.reduction M plans) h ready.storage schedule.programs initial (Preparation.outputHeap ready)
    let returned := ReturnedStorage.returned encode h views.output (caller codec views memory).indices
      (scratchRead h ready.storage native) before
    FPGrid.run M (FPScalarTree.reduction M plans) h schedule.programs initial=some (FPGrid.fold M (FPScalarTree.reduction M plans) h schedule.programs initial) ∧
    returned.result=() ∧ returned.outputObject=views.output.object ∧ returned.outputOwner=views.output.owner ∧
    (∀ p : Geometry.Program h,∀ n : Fin 256,
      (NativeGrid.outputPointer h ready.storage schedule abi p n).toNat=
        ready.storage.output.address (Geometry.outputOffset h p.batch p.row n.val)) ∧
    (∀ address,CallerViews.Owned views allocations address → native address=Preparation.outputHeap ready address) ∧
    (∀ i : Coord views.output,∀ byte : Fin 4,views.output.address i byte<2^64) ∧
    (∀ i : Coord views.output,Active h views.output (caller codec views memory).indices i → ∀ byte : Fin 4,
      returned.memory views.output.owner (views.output.address i byte)=
        encode views.output.negative
          (FPHost.reference M (FPScalarTree.reduction M plans) h (caller codec views memory) zero i.1.val
            ((caller codec views memory).indices i.1.val).toNat (i.2.val-h.slice)) byte) ∧
    (∀ i : Coord views.output,¬Active h views.output (caller codec views memory).indices i → ∀ byte : Fin 4,
      returned.memory views.output.owner (views.output.address i byte)=
        encode views.output.negative ((caller codec views memory).output i.1.val i.2.val) byte) ∧
    (∀ owner address,(¬∃ i : Coord views.output,∃ byte : Fin 4,
        Active h views.output (caller codec views memory).indices i ∧ owner=views.output.owner ∧
        address=views.output.address i byte) → returned.memory owner address=before owner address) := by
  exact complete M (FPScalarTree.reduction M plans) h primary schedule abi codec views memory zero
    packed other padding ready allocations viewFacts fresh guard encode before initialOutput valid

#axiomsClean complete_tree

#axiomsClean NativeGrid.output_readback
#axiomsClean NativeGrid.lane_words
#axiomsClean HostProtocol.prewrite_failure
#axiomsClean HostProtocol.empty_return
#axiomsClean HostProtocol.frame
#axiomsClean HostProtocol.success
#axiomsClean complete
#axiomsClean Preparation.source_loads
#axiomsClean PhysicalGrid.agrees
#axiomsClean NativeIndex.source_addresses
#axiomsClean NativeControl.cdiv_word
#axiomsClean ReturnedStorage.empty
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.FPEquiv

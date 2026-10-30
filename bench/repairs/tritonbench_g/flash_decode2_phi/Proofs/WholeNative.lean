import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.NativeSafety
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Numerical

namespace VeriTile.Bench.Repairs.FlashDecode2.WholeNative
open VeriTile.Triton FP.Structural Checked TensorBinding Native Numerical
noncomputable section

structure Result where
  returned : Option Unit
  output : VeriTile.Torch.FP32.Tensor
  memory : Physical.Bytes
  currentDevice : Nat
  launchDevice : Option Nat

def target (p : ScalarOps) (schedule : Schedule) (c : Call) (t : Tensors c allocations)
    (memory : Physical.Bytes) (row : Nat) (k : TileIndex [block c]) : Word :=
  eval (lower p) schedule (Grid.value algebra (entry t.names (Physical.load memory))
    c.masked (batches c) (heads c) (capacity c) (width c) c.blockSeq (block c) row k)

def invoke (p : ScalarOps) (schedule : Schedule) (c : Call) (t : Tensors c allocations)
    (memory : Physical.Bytes) : Except String Result :=
  if admitted : checkInput c (lengthValues t memory) = .ok () then
    let h := (check_input_iff c _).mp admitted
    .ok ⟨none,t.output.tensor,
      Physical.executeBytes (lower p) schedule t.names.out (request c h) (List.range (geometry c).rows)
        (entry t.names (Physical.load memory)) memory,
      t.callerDevice,if (geometry c).rows = 0 then none else some c.mid.metadata.device.index⟩
  else .error "input contract rejected"

structure Guarantee (actual reference : ScalarOps) (schedule : Schedule) (c : Call)
    (t : Tensors c allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory)) (result : Result) : Prop where
  returned : result.returned = none
  output : result.output = t.output.tensor
  storage : allocations t.names.out = result.output.storage
  shape : result.output.metadata.shape = [batches c,heads c,width c]
  dtype : result.output.metadata.dtype = .floating .fp32
  context : result.currentDevice = t.callerDevice
  launch : result.launchDevice = if (geometry c).rows = 0 then none else some c.mid.metadata.device.index
  addresses : NativeSafety.AllAccess c
  coordinates : NativeSafety.Coordinates c t
  arithmetic : NativeSafety.Arithmetic c t memory
  values : ∀ row (k : TileIndex [block c]),row < (geometry c).rows → lane k < width c → ∀ i : Fin 4,
    result.memory t.names.out (4*(row*width c+lane k)+i.val) =
      Physical.encode (target reference schedule c t memory row k) i
  frame : ∀ r i,r ≠ t.names.out ∨ 4*(geometry c).size ≤ i → result.memory r i = memory r i
  source : ∃ sourceResult,
    Host.invoke algebra (request c h) (nativePreparation c t memory h) (List.range (geometry c).rows) =
      some sourceResult ∧
    ∀ i,Physical.projectO (lower actual) schedule sourceResult.state i = result.memory t.names.out i

theorem tensor (actual reference : ScalarOps) (laws : ScalarLaws actual reference)
    (schedule : Schedule) (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory)) :
    ∃ result,invoke actual schedule c t memory = .ok result ∧
      Guarantee actual reference schedule c t memory h result := by
  cases laws.operations_eq actual reference
  let call := nativePreparation c t memory h
  have boundEntry : call.before = entry (bindings call) (Physical.load memory) := rfl
  obtain ⟨source,run,ret,out,context,launch,values,frame,related⟩ :=
    Physical.whole_bytes (lower actual) schedule (request c h) memory call boundEntry
      (List.range (geometry c).rows) (List.Perm.refl _)
  let answer : Result := ⟨none,t.output.tensor,
    Physical.executeBytes (lower actual) schedule t.names.out (request c h) (List.range (geometry c).rows)
      (entry t.names (Physical.load memory)) memory,
    t.callerDevice,if (geometry c).rows = 0 then none else some c.mid.metadata.device.index⟩
  refine ⟨answer,by simp only [invoke,dif_pos ((check_input_iff c _).mpr h)]; rfl,
    rfl,rfl,t.outputStorage,?_,t.output.laws.dtype,rfl,rfl,NativeSafety.checked_access c t memory h,
    NativeSafety.checked_coordinates c t h.metadata,NativeSafety.checked_arithmetic c t memory h,
    ?_,?_,⟨source,run,related⟩⟩
  · have shape := h.metadata.2.2.2.2.2.2.1
    rw [←t.outputObserved] at shape
    exact shape
  · intro row k hr hk i
    exact values row k hr hk i
  · intro r i outside
    exact frame r i outside

theorem empty (p : ScalarOps) (schedule : Schedule) (c : Call) (t : Tensors c allocations)
    (memory : Physical.Bytes) (h : InputContract c (lengthValues t memory))
    (empty : batches c = 0 ∨ heads c = 0) :
    invoke p schedule c t memory = .ok ⟨none,t.output.tensor,memory,t.callerDevice,none⟩ := by
  have zero : (geometry c).rows = 0 := by
    rcases empty with zero | zero <;> simp [geometry,Geometry.rows,zero]
  simp [invoke,(check_input_iff c _).mpr h,zero,Physical.executeBytes]

#axiomsClean tensor
#axiomsClean empty
#specNonCircular target avoiding [invoke,Host.invoke,Grid.source,Physical.executeBytes]
end
end VeriTile.Bench.Repairs.FlashDecode2.WholeNative

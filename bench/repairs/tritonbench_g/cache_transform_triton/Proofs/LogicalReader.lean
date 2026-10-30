import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.IOResult
import bench.repairs.tritonbench_g.cache_transform_triton.IO

namespace VeriTile.Bench.Repairs.CacheTransform.CacheIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

theorem byteAt_payload (M : Algebra ℝ) (p : Parameters) (n capacity : Nat)
    (xs : (cacheIO M p n capacity).Inputs) (row byte : Nat) (hb : byte<p.rowBytes) :
    (logicalInputsEquiv M p n capacity xs).byteAt row byte =
      payload M p n capacity xs (row*p.rowBytes+byte) := by
  by_cases hr : row<capacity
  · have hi : row*p.rowBytes+byte<capacity*p.rowBytes := by
      simpa [finProdFinEquiv,Nat.add_comm,Nat.mul_comm] using (finProdFinEquiv (⟨row,hr⟩,⟨byte,hb⟩)).isLt
    simp only [LogicalInputs.byteAt,hr,hb,and_self,dif_pos,logicalInputsEquiv,
      Equiv.coe_fn_mk,payload,hi]
    congr 2
    apply Fin.ext
    simp [finProdFinEquiv,Nat.add_comm,Nat.mul_comm]
  · have hi : ¬row*p.rowBytes+byte<capacity*p.rowBytes := by
      have := Nat.mul_le_mul_right p.rowBytes (Nat.le_of_not_gt hr)
      omega
    simp [LogicalInputs.byteAt,hr,payload,hi]

theorem logical_iff (M : Algebra ℝ) (p : Parameters) (n capacity : Nat) :
    (logicalIO M p n capacity ⊨{logicalDomain M p n capacity} fun xs j =>
      xs.byteAt (xs.rowAt (j.val/p.rowBytes)).toNat (j.val%p.rowBytes)) ↔
    (cacheIO M p n capacity).ImplementsOnCells (Domain M p n capacity)
      (formula M p n capacity) := by
  change (cacheIO M p n capacity).ImplementsOnCells
    (fun xs => logicalDomain M p n capacity (logicalInputsEquiv M p n capacity xs))
    (fun xs => (logicalOutputsEquiv M p n capacity).symm (fun j =>
      (logicalInputsEquiv M p n capacity xs).byteAt
        ((logicalInputsEquiv M p n capacity xs).rowAt (j.val/p.rowBytes)).toNat
        (j.val%p.rowBytes))) ↔ _
  have eq : (fun xs => (logicalOutputsEquiv M p n capacity).symm (fun j =>
      (logicalInputsEquiv M p n capacity xs).byteAt
        ((logicalInputsEquiv M p n capacity xs).rowAt (j.val/p.rowBytes)).toNat
        (j.val%p.rowBytes))) = formula M p n capacity := by
    funext xs o
    fin_cases o
    funext j
    change (logicalInputsEquiv M p n capacity xs).byteAt
      (row M p n capacity xs (j.1.val/p.rowBytes)).toNat (j.1.val%p.rowBytes) = _
    by_cases hw : p.rowBytes=0
    · simp [hw,LogicalInputs.byteAt,formula,payload]
    · exact byteAt_payload M p n capacity xs _ _ (Nat.mod_lt _ (Nat.pos_of_ne_zero hw))
  have deq : (fun xs => logicalDomain M p n capacity (logicalInputsEquiv M p n capacity xs)) =
      Domain M p n capacity := by
    funext xs
    simp only [logicalDomain,Equiv.symm_apply_apply]
  exact (congrArg₂ (cacheIO M p n capacity).ImplementsOnCells deq eq).to_iff

theorem logical_implements (M : Algebra ℝ) (p : Parameters) (n capacity : Nat)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit) :
    logicalIO M p n capacity ⊨{logicalDomain M p n capacity} fun xs j =>
      xs.byteAt (xs.rowAt (j.val/p.rowBytes)).toNat (j.val%p.rowBytes) :=
  (logical_iff M p n capacity).mpr (IOProof.implements M p start block grid n capacity)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.CacheTransform.CacheIO

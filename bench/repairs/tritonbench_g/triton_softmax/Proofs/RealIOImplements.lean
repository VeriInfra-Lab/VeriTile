import bench.repairs.tritonbench_g.triton_softmax.IO
import bench.repairs.tritonbench_g.triton_softmax.Proofs.RowIO

namespace VeriTile.Bench.Repairs.TritonSoftmax.RealIOImplements
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁
noncomputable section
open IO

theorem correct (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (positive : 0<B) (cover : n≤B)
    (injective : ∀ i j : Fin B,i.val<n → j.val<n → i.val*oc=j.val*oc → i=j) :
    (softmaxIO out inp ib ob ir ic orow oc n B ⊨ fun _ xs i =>
      Real.exp (xs i) / ∑ j : Fin B, if j.val<n then Real.exp (xs (j,PUnit.unit)) else 0) := by
  refine MaskedTileKernelIO₁.Implements.intro _ (RowIO.flattenOk ..) ?_ ?_
  · intro bounds s hin hout
    exact RowIO.safe out inp ib ob ir ic orow oc n B positive bounds s
      (fun i hi => hin (i,PUnit.unit) hi) (fun i hi => hout (i,PUnit.unit) hi)
  · intro s xs input
    obtain ⟨t,run,values,frame⟩ := RowIO.result out inp ib ob ir ic orow oc n B positive
      injective s (fun i => xs (i,PUnit.unit)) (fun i hi => input (i,PUnit.unit) hi)
    refine ⟨t,run,?_,?_⟩
    · rintro ⟨i,u⟩ hi
      cases u
      exact values i hi
    · intro r a miss
      apply frame r a
      intro i hi equal
      rcases miss with region | offset
      · exact region equal.1.symm
      · exact offset (i,PUnit.unit) hi equal.2.symm


#axiomsClean correct
end
end VeriTile.Bench.Repairs.TritonSoftmax.RealIOImplements

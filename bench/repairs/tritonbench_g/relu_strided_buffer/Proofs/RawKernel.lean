import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Kernel

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.Raw
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open StructuralTile
noncomputable section

/-- Canonical launch arguments constructed solely from raw signed views and
extent. No Tensor device, wrapper result or allocation metadata is an input. -/
def arguments (x y : SignedView.View) (n : Nat) : HostPlan.Arguments :=
  ⟨x,y,n,n,HostPlan.block n,CappedGrid.numCTAs n (HostPlan.block n),
    CappedGrid.tilesPerCTA n (HostPlan.block n),4,
    decide (CappedGrid.tilesPerCTA n (HostPlan.block n)=1),none⟩

theorem grid [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (x y : SignedView.View)
    {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : x.Valid n inputCapacity) (hout : y.Valid n outputCapacity) (hstride : y.stride≠0)
    (hsep : SignedGrid.FootprintsDisjoint x y n) (s : State α) :
    ∃ final,runGrid M (arguments x y n) s=some final ∧
      (∀ i,i<n → final.mem y.region (y.address i)=
        Cell.mk .fp32 (scalarResult M cmp ((s.mem x.region (x.address i)).read .fp32))) ∧
      (∀ i,i<n → final.mem x.region (x.address i)=s.mem x.region (x.address i)) ∧
      (∀ r a,(r≠y.region ∨ ∀ i,i<n → a≠y.address i) → final.mem r a=s.mem r a) ∧
      (∀ i,i<n → x.address i < inputCapacity ∧ y.address i < outputCapacity) := by
  exact grid_result M cmp hc
    (⟨x.region,x.base,[n],[x.stride],none⟩ : HostPlan.Buffer)
    (⟨y.region,y.base,[n],[y.stride],none⟩ : HostPlan.Buffer)
    hn x.stride y.stride hin hout hstride hsep s

#axiomsClean grid
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.Raw

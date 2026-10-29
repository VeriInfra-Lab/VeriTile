import bench.tritonbench_g.rotary_emb.HostCorrect

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
open VeriTile Triton FullMath
noncomputable section

/-- Noncontiguous 17-sequence, five-Q-head, ten-K-head tensors, with an odd
rotary width of five inside full width seven. Both grid axes have two blocks. -/
def witnessRequest : Request :=
  ⟨⟨17,5,7,100,16,2⟩,⟨17,10,7,200,16,2⟩,
   ⟨17,7,20,2⟩,⟨17,7,30,3⟩,0.75⟩

def witnessPlan : Plan := ⟨witnessRequest,5⟩
def witnessAlloc : FlatAlloc :=
  FlatAlloc.ofList "flat" [("Q",4000),("K",4000),("Cos",1000),("Sin",1000)]

theorem witness_domain : Domain witnessPlan witnessAlloc.extent "Q" "K" "Cos" "Sin" := by
  constructor
  · decide
  · decide
  · exact regions_nonempty
  · intro x; cases x <;> decide
  · intro x
    cases x <;> intro n h d n' h' d' hn hh hd hn' hh' hd' he <;>
      dsimp [Plan.tensor,witnessPlan,witnessRequest,Tensor3.address] at *
    all_goals
      have en : n=n' := by omega
      subst n'
      have eh : h=h' := by omega
      subst h'
      exact ⟨rfl,rfl,by omega⟩
  · intro x n h d hn hh hgrid hd
    cases x <;>
      simp [Plan.tensor,witnessPlan,witnessRequest,Tensor3.address,Plan.evenWidth,
        Plan.parameters,region,witnessAlloc,FlatAlloc.listExtent,Plan.headBlocks] at * <;> omega
  · intro n d hn hd
    simp [witnessPlan,witnessRequest,Plan.evenWidth,Plan.parameters,witnessAlloc,
      FlatAlloc.listExtent] at *
    omega

theorem witness_exists (q k cos sin : FullSource.Storage) :
    ∃ t, witnessPlan.run (witnessPlan.kernel q k cos sin "Q" "K" "Cos" "Sin")
      witnessPlan.schedule default = some t ∧
      witnessPlan.Output "Q" "K" "Cos" "Sin" default t := by
  exact (grid_correct witnessPlan q k cos sin "Q" "K" "Cos" "Sin"
    witnessAlloc (FlatAlloc.ofList_disjoint _ _ (by decide)) (FlatAlloc.ofList_closed _ _)
    witness_domain (by simp [witnessAlloc]) (by simp [witnessAlloc]) default).2

theorem witness_grid_nonempty : witnessPlan.schedule.length=4 := by decide
theorem witness_k_extra_heads : witnessPlan.headBlocks*4=8 ∧ witnessRequest.k.heads=10 := by decide
theorem witness_odd_partial_tail : witnessPlan.evenWidth=4 ∧ witnessRequest.q.width=7 := by decide

#axiomsClean witness_domain
#axiomsClean witness_exists
#axiomsClean witness_grid_nonempty
#axiomsClean witness_k_extra_heads
#axiomsClean witness_odd_partial_tail

end
end VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid

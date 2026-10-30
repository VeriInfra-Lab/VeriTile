import bench.tritonbench_g.chunk_gla_simple.Proofs.Admission
import bench.tritonbench_g.chunk_gla_simple.Proofs.DirectReal

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Static
open VeriTile Triton
open Host
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
noncomputable section
set_option maxHeartbeats 8000000

/-- The original BT domain already provides the third dot's K lower bound.
BK16 fixes both QH/QK dot inner dimensions without changing K or V. -/
theorem dot_dimensions (c : Shape) (h : c.BT∈Admission.blocks) :
    16≤c.BT ∧ 16≤c.BK ∧ 1≤c.BV ∧ c.BK≤64 ∧ c.BV≤64 := by
  have bt : 16≤c.BT := by
    simp only [Admission.blocks,List.mem_cons,List.not_mem_nil,or_false] at h
    rcases h with h|h|h|h|h|h|h <;> omega
  exact ⟨bt,c.bk_min,block_pos c.V,c.bk_max,Nat.min_le_right _ _⟩

/-- Maximum source tensor size equals, and never exceeds, Triton's 2^20
inclusive static limit. No tighter BT, K or V guard is introduced. -/
theorem tensor_sizes (c : Shape) (h : c.BT∈Admission.blocks) :
    c.BT*c.BT≤1048576 ∧ c.BT*c.BK≤65536 ∧ c.BT*c.BV≤65536 ∧ c.BK*c.BV≤4096 := by
  have ht:=(Admission.block_bounds c h).2
  have hk:=c.bk_max
  have hv : c.BV≤64 := Nat.min_le_right _ _
  exact ⟨by nlinarith,by nlinarith,by nlinarith,by nlinarith⟩

theorem k_zero_loop (c : Shape) (hk : c.K=0) : numBlocks (layout c)=0 := by
  change (c.K+c.BK-1)/c.BK=0
  rw [hk,Nat.zero_add]
  exact Nat.div_eq_of_lt (by have hp:=c.bk_pos; omega)

/-- K0 remains a successful zero-trip source branch. The Real target is zero;
no analogous universal FP zero law is asserted for exceptional gate words. -/
theorem k_zero_real (c : Shape) (bt : 0<c.BT) (hk : c.K=0)
    (a : Inputs) (scale : ℝ) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    Host.formula c bt a scale h t v=0 := by
  simp [Host.formula,expected,score,qhSum,qkSum,k_zero_loop c hk]

def witness : Shape := ⟨1,1,1,1,1,16⟩
def ones : Inputs := ⟨fun _=>1,fun _=>1,fun _=>1,fun _=>1,fun _=>0⟩

/-- Smallest original failure now has a nonvacuous independent Real value. -/
theorem witness_two : Host.formula witness (by decide) ones 1 ⟨0,by decide⟩ ⟨0,by decide⟩ ⟨0,by decide⟩=2 := by
  change expected ⟨1,1,1,1,1,1,1,1,1,16,16,1⟩ ⟨0,0,0⟩ ones 1 0 0=2
  norm_num [expected,qhSum,qkSum,score,query,key,history,value,gate,active,
    keyAddress,historyAddress,outputAddress,gateAddress,row,column,numBlocks,ones,
    Fin.sum_univ_succ]

#axiomsClean dot_dimensions
#axiomsClean tensor_sizes
#axiomsClean k_zero_real
#axiomsClean witness_two
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Static

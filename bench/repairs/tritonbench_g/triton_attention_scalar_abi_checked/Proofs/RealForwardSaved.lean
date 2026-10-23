import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealForwardCTA
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source Mathematical
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem forward_saved (c : Shape) (scale : ℝ) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads (forwardParameters c scale) s μ)
    (h0 : s.pids 0<c.blocks) (h1 : s.pids 1<c.groups)
    (i : FRow (forwardParameters c scale)) :
    ∃ m,forwardMCell model (forwardParameters c scale) s i=.mk .fp32 (some m) ∧
      forwardLCell model (forwardParameters c scale) s i=.mk .fp32 (some
        (mass (s.pids 0*128+i.1.val) c.rows (logicalScore (forwardParameters c scale) s μ i.1) m)) := by
  let lane : FQuery (forwardParameters c scale):=(i.1,⟨0,c.dim_pos⟩,())
  have hb : ∀ b<forwardBlocks (forwardParameters c scale) s, ∀ j : Fin 128,
      forwardHeadRow (forwardParameters c scale) s+b*128+j.val<c.groups*c.rows := by
    intro b hh j
    rw [forward_head]
    rw [forward_blocks] at hh
    have hj:=j.isLt
    have hr : b*128+j.val<c.rows := by change _<c.blocks*128;omega
    have hg:=Nat.mul_le_mul_right c.rows (Nat.succ_le_of_lt h1)
    nlinarith
  have hc : ∀ b<forwardBlocks (forwardParameters c scale) s,
      b*128 ≤ forwardQueryRow (forwardParameters c scale) s i.1 := by
    intro b hh
    rw [forward_blocks] at hh
    dsimp [forwardQueryRow,forwardParameters]
    omega
  have hn : forwardQueryRow (forwardParameters c scale) s i.1+1 ≤
      forwardBlocks (forwardParameters c scale) s*128 := by
    rw [forward_blocks]
    have hi:=i.1.isLt
    change i.1.val<128 at hi
    dsimp [forwardQueryRow,forwardParameters]
    omega
  have hrows : s.pids 0*128+i.1.val+1 ≤ c.rows := by
    have hi:=i.1.isLt
    change i.1.val<128 at hi
    change _≤c.blocks*128
    omega
  obtain ⟨m,hm,hl,_⟩:=loop_consistent (forwardParameters c scale) s μ h lane (by change 0<128;decide)
    (forwardBlocks (forwardParameters c scale) s) (by rw [forward_blocks];omega) hb hc
  have him : (i.1,PUnit.unit)=i := by exact Prod.ext rfl (Subsingleton.elim _ _)
  dsimp only [lane] at hm hl
  rw [him] at hm hl
  refine ⟨m,?_,?_⟩
  · simp only [forwardMCell,hm];rfl
  · simp only [forwardLCell,hl]
    change Cell.mk .fp32 (some (mass (s.pids 0*128+i.1.val)
      (forwardBlocks (forwardParameters c scale) s*128) (logicalScore (forwardParameters c scale) s μ i.1) m))=_
    change s.pids 0*128+i.1.val+1 ≤ forwardBlocks (forwardParameters c scale) s*128 at hn
    rw [mass_done _ _ _ _ hn,mass_done _ _ _ _ hrows]
    rfl

#axiomsClean forward_saved
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased

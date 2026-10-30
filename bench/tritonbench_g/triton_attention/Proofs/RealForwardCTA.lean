import bench.tritonbench_g.triton_attention.Proofs.RealForwardInvariant
import bench.tritonbench_g.triton_attention.Kernels
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem forward_head (c : Shape) (scale : ℝ) (s : State R) :
    forwardHeadRow (forwardParameters c scale) s=s.pids 1*c.rows := by
  simp [forwardHeadRow,forwardParameters,Nat.mul_div_cancel,c.dim_pos]
theorem forward_blocks (c : Shape) (scale : ℝ) (s : State R) :
    forwardBlocks (forwardParameters c scale) s=s.pids 0+1 := by
  simp [forwardBlocks,forwardStop,forwardParameters]
theorem forward_offset_injective (p : ForwardParameters) (s : State R) :
    Function.Injective (forwardOffset p s) := by
  intro a b he
  have ha:=a.2.1.isLt
  have hb:=b.2.1.isLt
  have hcol : a.2.1=b.2.1 := by
    apply Fin.ext
    have hh:=congrArg (· % p.head) he
    simpa only [forwardOffset,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt ha,Nat.mod_eq_of_lt hb] using hh
  have hdim : 0<p.head := by omega
  have hrow : a.1=b.1 := by
    apply Fin.ext
    simp only [forwardOffset,forwardQueryRow] at he
    have hc:=congrArg Fin.val hcol
    nlinarith
  exact Prod.ext hrow (Prod.ext hcol (Subsingleton.elim _ _))

theorem forward_answer (c : Shape) (scale : ℝ) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads (forwardParameters c scale) s μ)
    (h0 : s.pids 0<c.blocks) (h1 : s.pids 1<c.groups)
    (i : FQuery (forwardParameters c scale)) :
    forwardOutput model (forwardParameters c scale) s i = .mk .fp32 (some
      (Mathematical.attention c.dim scale
        (qAt (forwardParameters c scale) s μ) (kAt (forwardParameters c scale) s μ)
        (vAt (forwardParameters c scale) s μ) (s.pids 0*128+i.1.val) i.2.1.val)) := by
  have hb : ∀ b<forwardBlocks (forwardParameters c scale) s, ∀ j : Fin 128,
      forwardHeadRow (forwardParameters c scale) s+b*128+j.val<c.groups*c.rows := by
    intro b hh j
    rw [forward_head]
    rw [forward_blocks] at hh
    have hj:=j.isLt
    have hr : b*128+j.val<c.rows := by
      change b*128+j.val<c.blocks*128
      omega
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
  have ans:=answer_real (forwardParameters c scale) s μ h i (by change 0<128; decide) hb hc hn
  simp only [forwardOutput,ans]
  rfl

/-- A full shared-source forward CTA, with its actual output store and frame. -/
theorem forward_execution_real (c : Shape) (scale : ℝ) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads (forwardParameters c scale) s μ)
    (h0 : s.pids 0<c.blocks) (h1 : s.pids 1<c.groups) :
    ∃ final,FP.Structural.exec (backend algebra model) (forwardKernel c scale) s=some final ∧
      (∀ i : FQuery (forwardParameters c scale),final.mem "Out" (forwardOffset (forwardParameters c scale) s i)=
        .mk .fp32 (some (Mathematical.attention c.dim scale
          (qAt (forwardParameters c scale) s μ) (kAt (forwardParameters c scale) s μ)
          (vAt (forwardParameters c scale) s μ) (s.pids 0*128+i.1.val) i.2.1.val))) ∧
      (∀ (r : RegionName) o,r≠"Out" → r≠"L" → r≠"M" → final.mem r o=s.mem r o) := by
  obtain ⟨final,he,_,ho,_,_,hf⟩:=forwardCorrectness algebra model (forwardParameters c scale) s
    (by change 0<128; decide) (by dsimp [forwardStop,forwardParameters]; exact dvd_mul_left 128 _)
    (by simp [forwardParameters]) (by simp [forwardParameters]) (by simp [forwardParameters]) (forward_offset_injective _ _)
  refine ⟨final,he,?_,?_⟩
  · intro i
    apply Eq.trans (ho i ?_) (forward_answer c scale s μ h h0 h1 i)
    simp only [forwardOutPointer,forwardQPointer,forward_head,
      TileShape.blockPtr_inBounds_2d_offsets_index,Nat.zero_add,forwardParameters]
    have hi:=i.1.isLt
    have hd:=i.2.1.isLt
    change i.1.val<128 at hi
    change i.2.1.val<c.dim at hd
    have hr : s.pids 0*128+i.1.val<c.rows := by change _<c.blocks*128; omega
    have hg:=Nat.mul_le_mul_right c.rows (Nat.succ_le_of_lt h1)
    have hh : s.pids 1*c.rows+s.pids 0*128+i.1.val<c.groups*c.rows := by nlinarith
    simpa [forwardHeadRow,c.dim_pos,hd] using hh
  · intro r o ho hl hm
    exact hf r o (Or.inl ho) (Or.inl hl) (Or.inl hm)

#axiomsClean forward_execution_real
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased

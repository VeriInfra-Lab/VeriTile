import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.rmsnorm_fused.Proofs.FullContracts
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.RmsnormFused.Real
open VeriTile Triton
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Independent mathematical RMSNorm, indexed by every logical row/column. -/
def formula (s : BlockState) (X W : RegionName) (row N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  s.readMem X (row*N+k) *
    (Real.sqrt ((∑ j : Fin N,s.readMem X (row*N+j.val)^2)/(N:ℝ)+eps))⁻¹ * s.readMem W k

def setup (M row : Nat) (s : BlockState) : BlockState :=
  {s with pids := fun a => if a=0 then row else 0
          numPids := fun a => if a=0 then M else 1}

theorem projected_exec (X Y W : RegionName) (N B : Nat) (eps : ℝ) (s : BlockState) :
    exec (kernel X Y W N B eps).eraseDType.toAlgKernel s =
      exec (VeriTile.Bench.TritonBenchG.RmsnormFused.rms_norm_fwd_fused X Y W N N B eps).toAlgKernel s := by
  change exec (VeriTile.Bench.TritonBenchG.RmsnormFused.typedKernel .fp32 .fp32 .fp32
    Bool.false X Y W N N B eps).eraseDType.toAlgKernel s = _
  rw [VeriTile.Bench.TritonBenchG.RmsnormFused.typed_projection]
  rfl

def source (X Y W : RegionName) (M N B : Nat) (eps : ℝ) :
    List Nat → BlockState → Option BlockState
  | [],s => some s
  | r::rest,s => (exec (kernel X Y W N B eps).eraseDType.toAlgKernel (setup M r s)).bind
      (source X Y W M N B eps rest)

def Outside (Y : RegionName) (N : Nat) (rows : List Nat) (R : RegionName) (o : Nat) : Prop :=
  R≠Y ∨ ∀ r∈rows,∀ k,k<N → o≠r*N+k

theorem formula_congr (s t : BlockState) (X W : RegionName) (row N : Nat) (eps : ℝ)
    (k : Nat) (hx : ∀ o,s.mem X o=t.mem X o) (hw : ∀ o,s.mem W o=t.mem W o) :
    formula s X W row N eps k=formula t X W row N eps k := by
  simp [formula,BlockState.readMem,hx,hw]

/-- Reuses the original proved complete two-pass execution. The new induction
composes every actual state, every logical output and every outside cell. -/
theorem complete (X Y W : RegionName) (M N B : Nat) (eps : ℝ)
    (hB : 0<B) (hXY : X≠Y) (hWY : W≠Y) (rows : List Nat) (unique : rows.Nodup)
    (s : BlockState) :
    ∃ t,source X Y W M N B eps rows s=some t ∧
      (∀ r∈rows,∀ k,k<N → t.readMem Y (r*N+k)=formula s X W r N eps k) ∧
      (∀ R o,Outside Y N rows R o → t.mem R o=s.mem R o) := by
  induction rows generalizing s with
  | nil => exact ⟨s,rfl,by simp,by intro R o h; rfl⟩
  | cons r rest ih =>
    have un := List.nodup_cons.mp unique
    obtain ⟨mid,he,hv,hf⟩ := VeriTile.Bench.TritonBenchG.RmsnormFused.General.real_execution
      (setup M r s) X Y W N N B eps hB hXY hWY
    have execution : exec (kernel X Y W N B eps).eraseDType.toAlgKernel (setup M r s)=some mid :=
      (projected_exec X Y W N B eps _).trans he
    have frame : ∀ R o,(R≠Y ∨ ∀ k,k<N → o≠r*N+k) → mid.mem R o=s.mem R o := by
      intro R o ho
      apply hf R o
      exact ho.imp_right (fun h k => h k.val k.isLt)
    have vals : ∀ k,k<N → mid.readMem Y (r*N+k)=formula s X W r N eps k := by
      intro k hk
      simpa [formula,setup,VeriTile.Bench.TritonBenchG.RmsnormFused.General.rowSpec]
        using hv ⟨k,hk⟩
    obtain ⟨t,ht,vt,ft⟩ := ih un.2 mid
    refine ⟨t,by simpa [source,execution] using ht,?_,?_⟩
    · intro q hq k hk
      by_cases inside : q∈rest
      · rw [vt q inside k hk]
        exact formula_congr mid s X W q N eps k
          (fun o => frame X o (.inl hXY)) (fun o => frame W o (.inl hWY))
      · have same : q=r := (List.mem_cons.mp hq).resolve_right inside
        subst q
        have untouched := ft Y (r*N+k) (Or.inr (by
          intro q hq j hj
          exact Row.row_offset_ne N r q k j hk hj (by intro eq; subst q; exact un.1 hq)))
        unfold BlockState.readMem
        rw [untouched]
        exact vals k hk
    · intro R o ho
      rw [ft R o (ho.imp_right (fun h q hq => h q (by simp [hq])))]
      exact frame R o (ho.imp_right (fun h k hk => h r (by simp) k hk))

theorem whole (s : Shape) (valid : s.Valid) (X Y W : RegionName) (eps : ℝ)
    (hXY : X≠Y) (hWY : W≠Y) (rows : List Nat) (all : rows.Perm s.grid) (m : BlockState) :
    ∃ t,source X Y W s.rows s.width s.block eps rows m=some t ∧
      (∀ r k,r<s.rows → k<s.width → t.readMem Y (s.offset r k)=formula m X W r s.width eps k) ∧
      (∀ R o,(R≠Y ∨ s.rows*s.width≤o) → t.mem R o=m.mem R o) := by
  obtain ⟨t,ht,hv,hf⟩ := complete X Y W s.rows s.width s.block eps
    (block_positive s valid) hXY hWY rows (all.nodup_iff.mpr (grid_unique s)) m
  refine ⟨t,ht,?_,?_⟩
  · intro r k hr hk
    exact hv r (all.mem_iff.mpr ((grid_complete s r).mpr hr)) k hk
  · intro R o ho
    apply hf
    rcases ho with h|h
    · exact Or.inl h
    · right
      intro r hr k hk eq
      have bound := offset_bounded s r k ((grid_complete s r).mp (all.mem_iff.mp hr)) hk
      change r*s.width+k<s.rows*s.width at bound
      omega

#axiomsClean projected_exec
#axiomsClean complete
#axiomsClean whole
#specNonCircular formula avoiding [source,kernel,exec]
end
end VeriTile.Bench.Repairs.RmsnormFused.Real

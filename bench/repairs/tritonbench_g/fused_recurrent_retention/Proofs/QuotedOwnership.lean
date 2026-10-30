import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedStoreGeometry

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
noncomputable section

/-- Actual source output pointers inherit unique writers, including masked
tails. This connects the existing grid ownership proof to quoted stores. -/
theorem output_writer (s : Shape) (b : InputBuffers) (domain : StoreDomain s b)
    (p p' : ProgramId) (t t' lv lv' : Nat) (hp : p.Valid s) (hp' : p'.Valid s)
    (ht : t<s.T) (ht' : t'<s.T) (hl : lv<s.BV) (hl' : lv'<s.BV)
    (hv : p.value s lv<s.DV) (hv' : p'.value s lv'<s.DV)
    (he : outputOffset (layout s b) p t lv=outputOffset (layout s b) p' t' lv') :
    p=p' ∧ t=t' ∧ lv=lv' := by
  rw [outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1,
    outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1] at he
  exact output_writer_unique s p p' t t' lv lv' hp hp' ht ht' hl hl' hv hv' he

theorem gradient_writer (s : Shape) (b : InputBuffers) (domain : StoreDomain s b)
    (p p' : ProgramId) (t t' lk lk' : Nat) (hp : p.Valid s) (hp' : p'.Valid s)
    (ht : t<s.T) (ht' : t'<s.T) (hl : lk<s.BK) (hl' : lk'<s.BK)
    (hk : p.key s lk<s.DK) (hk' : p'.key s lk'<s.DK)
    (he : gradientOffset (layout s b) p t lk=gradientOffset (layout s b) p' t' lk') :
    p=p' ∧ t=t' ∧ lk=lk' := by
  rw [gradientOffset_partial s b domain.qDense (by omega) domain.shape.1,
    gradientOffset_partial s b domain.qDense (by omega) domain.shape.1] at he
  exact gradient_writer_unique s p p' t t' lk lk' hp hp' ht ht' hl hl' hk hk' he

theorem final_writer (s : Shape) (b : InputBuffers) (p p' : ProgramId) (lk lk' lv lv' : Nat)
    (hlk : lk<s.BK) (hlk' : lk'<s.BK) (hlv : lv<s.BV) (hlv' : lv'<s.BV)
    (hk : p.key s lk<s.DK) (hk' : p'.key s lk'<s.DK)
    (hv : p.value s lv<s.DV) (hv' : p'.value s lv'<s.DV)
    (he : seedOffset (layout s b) p lk lv=seedOffset (layout s b) p' lk' lv') :
    p=p' ∧ lk=lk' ∧ lv=lv' := by
  rw [seedOffset_state,seedOffset_state] at he
  exact final_writer_unique s p p' lk lk' lv lv' hlk hlk' hlv hlv' hk hk' hv hv' he

theorem reverse_gradient_writer (s : Shape) (b : InputBuffers) (domain : StoreDomain s b)
    (p p' : ProgramId) (r r' lk lk' : Nat) (hp : p.Valid s) (hp' : p'.Valid s)
    (hr : r<s.T) (hr' : r'<s.T) (hl : lk<s.BK) (hl' : lk'<s.BK)
    (hk : p.key s lk<s.DK) (hk' : p'.key s lk'<s.DK)
    (he : reverseGradientOffset (layout s b) p r lk=reverseGradientOffset (layout s b) p' r' lk') :
    p=p' ∧ r=r' ∧ lk=lk' := by
  rw [reverseGradient_at _ _ _ _ hr,reverseGradient_at _ _ _ _ hr'] at he
  obtain ⟨hpid,ht,hlane⟩ := gradient_writer s b domain p p' _ _ lk lk' hp hp'
    (by dsimp [layout]; omega) (by dsimp [layout]; omega) hl hl' hk hk' he
  exact ⟨hpid,by dsimp [layout] at ht; omega,hlane⟩

theorem reverse_output_writer (s : Shape) (b : InputBuffers) (domain : StoreDomain s b)
    (p p' : ProgramId) (r r' lv lv' : Nat) (hp : p.Valid s) (hp' : p'.Valid s)
    (hr : r<s.T) (hr' : r'<s.T) (hl : lv<s.BV) (hl' : lv'<s.BV)
    (hv : p.value s lv<s.DV) (hv' : p'.value s lv'<s.DV)
    (he : reverseOutputOffset (layout s b) p r lv=reverseOutputOffset (layout s b) p' r' lv') :
    p=p' ∧ r=r' ∧ lv=lv' := by
  rw [reverseOutput_at _ _ _ _ hr,reverseOutput_at _ _ _ _ hr'] at he
  obtain ⟨hpid,ht,hlane⟩ := output_writer s b domain p p' _ _ lv lv' hp hp'
    (by dsimp [layout]; omega) (by dsimp [layout]; omega) hl hl' hv hv' he
  exact ⟨hpid,by dsimp [layout] at ht; omega,hlane⟩

#axiomsClean output_writer
#axiomsClean gradient_writer
#axiomsClean final_writer
#axiomsClean reverse_gradient_writer
#axiomsClean reverse_output_writer
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted

import bench.tritonbench_g.lightning_attention.Proofs.FPExecution
import bench.tritonbench_g.lightning_attention.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem rowMajor_injective (a b stride rowBase base colBase : Nat) (hb : b ≤ stride) :
    Function.Injective (fun i : TileIndex [a,b] => base+(rowBase+i.1.val)*stride+colBase+i.2.1.val) := by
  rintro ⟨⟨ra,hra⟩,⟨ca,hca⟩,⟨⟩⟩ ⟨⟨rb,hrb⟩,⟨cb,hcb⟩,⟨⟩⟩ h
  dsimp only at h
  have step : ∀ x y : Nat, x<y → (rowBase+x)*stride+stride ≤ (rowBase+y)*stride := by
    intro x y hxy
    have hm : (rowBase+x+1)*stride ≤ (rowBase+y)*stride := Nat.mul_le_mul_right stride (by omega)
    simpa [Nat.add_mul] using hm
  have hr : ra=rb := by
    rcases Nat.lt_trichotomy ra rb with hlt|heq|hgt
    · have hs := step ra rb hlt; omega
    · exact heq
    · have hs := step rb ra hgt; omega
  subst rb
  have hc : ca=cb := by omega
  subst cb
  rfl

theorem kv_layout (p : Parameters) : Function.Injective (kvOffset p) := by
  simpa [kvOffset] using rowMajor_injective p.keyDim p.columns p.columns 0 0 0 (Nat.le_refl _)

theorem inter_layout (p : Parameters) : Function.Injective (interOffset p) := by
  simpa [interOffset] using rowMajor_injective p.rows p.columns p.columns 0 0 0 (Nat.le_refl _)

theorem body_layout {α : Type} (p : Parameters) (s : State α) (h : p.columns ≤ p.valueDim) :
    Function.Injective (bodyOffset p s) :=
  rowMajor_injective p.rows p.columns p.valueDim (p.blockIndex*p.rows)
    (s.pids 0*p.length*p.valueDim) (s.pids 1*p.columns) h

theorem written_post {α : Type} {shape : TileShape} (output : RegionName) (offset : TileIndex shape → Nat)
    (active : TileIndex shape → Bool) (v : TileIndex shape → α) (s : State α)
    (hi : Function.Injective offset) : Post output offset active v s (written output offset active v s) := by
  constructor
  · intro i ha
    exact State.masked_scatter_readback _ _ _ (fun i => active i=Bool.true) s i ha hi
  · intro r o h
    exact State.masked_scatter_frame _ _ _ (fun i => active i=Bool.true) _ r o
      (h.imp_right fun hh i _ => hh i) s

theorem kv_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec (backend M D) (kvSource p) s=some final ∧
      Post p.output (kvOffset p) (fun _ => Bool.true) (kvValues M D p s) s final := by
  obtain ⟨final,he,hm⟩ := kv_execution_memory M D p s
  refine ⟨final,he,?_⟩
  have hp := written_post p.output (kvOffset p) (fun _ => Bool.true) (kvValues M D p s) s (kv_layout p)
  simpa only [Post,hm] using hp
#axiomsClean kv_correctness

theorem kv_software_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) D) (kvSource p) s=some final ∧
      Post p.output (kvOffset p) (fun _ => Bool.true) (wordKV cfg D p s) s final := by
  simpa only [kv_software] using kv_correctness (software rest cfg) D p s
#axiomsClean kv_software_correctness

theorem inter_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec (backend M D) (interSource p) s=some final ∧
      Post p.output (interOffset p) (fun _ => Bool.true) (interValues M D p s) s final := by
  obtain ⟨final,he,hm⟩ := inter_execution_memory M D p s
  refine ⟨final,he,?_⟩
  have hp := written_post p.output (interOffset p) (fun _ => Bool.true) (interValues M D p s) s (inter_layout p)
  simpa only [Post,hm] using hp
#axiomsClean inter_correctness

theorem inter_software_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) D) (interSource p) s=some final ∧
      Post p.output (interOffset p) (fun _ => Bool.true) (wordInter D p s) s final := by
  simpa only [inter_software] using inter_correctness (software rest cfg) D p s
#axiomsClean inter_software_correctness

theorem body_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (hwidth : p.columns ≤ p.valueDim) :
    ∃ final, FP.Structural.exec (backend M D) (bodySource p) s=some final ∧
      Post p.output (bodyOffset p s) (bodyActive p) (bodyValues M D p s) s final := by
  obtain ⟨final,he,hm⟩ := body_execution_memory M D p s
  refine ⟨final,he,?_⟩
  have hp := written_post p.output (bodyOffset p s) (bodyActive p) (bodyValues M D p s) s (body_layout p s hwidth)
  simpa only [Post,hm] using hp
#axiomsClean body_correctness

theorem body_software_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) (hwidth : p.columns ≤ p.valueDim) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) D) (bodySource p) s=some final ∧
      Post p.output (bodyOffset p s) (bodyActive p) (wordBody cfg D p s) s final := by
  simpa only [body_software] using body_correctness (software rest cfg) D p s hwidth
#axiomsClean body_software_correctness

theorem grad_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (hinj : Function.Injective (gradOffset p s)) :
    ∃ final, FP.Structural.exec (backend M D) (gradSource p) s=some final ∧
      Post p.output (gradOffset p s) (gradActive p s) (gradValues M p s) s final := by
  obtain ⟨final,he,hm⟩ := grad_execution_memory M D p s
  refine ⟨final,he,?_⟩
  have hp := written_post p.output (gradOffset p s) (gradActive p s) (gradValues M p s) s (hinj)
  simpa only [Post,hm] using hp
#axiomsClean grad_correctness

theorem grad_software_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) (hinj : Function.Injective (gradOffset p s)) :
    ∃ final, FP.Structural.exec (backend (software rest cfg) D) (gradSource p) s=some final ∧
      Post p.output (gradOffset p s) (gradActive p s) (wordGrad p s) s final := by
  simpa only [grad_software] using grad_correctness (software rest cfg) D p s hinj
#axiomsClean grad_software_correctness

end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPValue

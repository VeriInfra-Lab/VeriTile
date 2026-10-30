import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPComplete
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue
open VeriTile.Triton FP.Structural
noncomputable section

/-- Optional old clean-undef model face. The main theorem does not impose this
policy, and the other-less Python loads do not establish it on native hardware. -/
def zeroUndefState {α : Type} (M : Algebra α) (s : State α) : State α :=
  {s with undef := fun d r o => match d with | .fp32 => M.literal (some .fp32) .real 0 | d => s.undef d r o}

theorem zeroUndefState_eq {α : Type} (M : Algebra α) (s : State α)
    (hz : ∀ r o, s.undef .fp32 r o=M.literal (some .fp32) .real 0) :
    zeroUndefState M s=s := by
  unfold zeroUndefState
  congr 1
  funext d r o
  cases d <;> simp [hz]

def zeroPaddingValue {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) : CIndex p → α :=
  outputValue M D p (zeroUndefState M s)

/-- The zero-undef face still evaluates every cast, multiplication/dot and add;
zero times infinity and signed-zero behavior are never simplified away. -/
theorem clean_undef_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α)
    (hz : ∀ r o, initial.undef .fp32 r o=M.literal (some .fp32) .real 0)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial = some final ∧
      Post p initial final (zeroPaddingValue M D p initial) := by
  obtain ⟨t,he,hm,hp⟩ := correctness M D p initial hw
  exact ⟨t,he,by simpa [zeroPaddingValue,zeroUndefState_eq M initial hz] using hp⟩

#axiomsClean zeroUndefState_eq
#axiomsClean clean_undef_correctness
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue

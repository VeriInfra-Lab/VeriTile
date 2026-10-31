import bench.tritonbench_g.rbe_triton_transform.Proofs.Source
namespace VeriTile.Bench.Repairs.RbeTritonTransform.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

/-- The two masks are intentionally distinct. Odd trailing dimensions can
read an implicit zero for the imaginary component while writing the real one. -/
def WritesApart {α : Type} (p : Parameters) (s : State α) : Prop :=
  (∀ imag i j, active p s imag i → active p s imag j →
    outputOffset p s imag i = outputOffset p s imag j → i = j) ∧
  (∀ i j, active p s Bool.false i → active p s Bool.true j →
    outputOffset p s Bool.false i ≠ outputOffset p s Bool.true j)

def Post {α : Type} (p : Parameters) (initial final : State α)
    (results : Bool → Index p → α) : Prop :=
  (∀ imag i, active p initial imag i →
    final.mem p.output (outputOffset p initial imag i) = .mk .fp32 (results imag i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ imag i, active p initial imag i →
    o ≠ outputOffset p initial imag i) → final.mem r o = initial.mem r o)

theorem written_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hw : WritesApart p s) (imag : Bool) (i : Index p) (hi : active p s imag i) :
    (written M p s).mem p.output (outputOffset p s imag i) = .mk .fp32 (value M p s imag i) := by
  cases imag
  · unfold written halfScatter
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inr (by
      intro j _ hj
      exact hw.2 i j hi hj))]
    exact masked_readback_active _ _ _ _ _ i hi (hw.1 Bool.false)
  · exact masked_readback_active _ _ _ _ _ i hi (hw.1 Bool.true)

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ imag i, active p s imag i → o ≠ outputOffset p s imag i) :
    (written M p s).mem r o = s.mem r o := by
  unfold written halfScatter
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _ hi => h Bool.true i hi))]
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _ hi => h Bool.false i hi))]

#axiomsClean written_readback
#axiomsClean written_frame
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Source

import bench.tritonbench_g.quantize_global.IO
import bench.tritonbench_g.quantize_global.Proofs.ReaderBefore
namespace QuantizeGlobalRepair.LogicalReader
open VeriTile VeriTile.Triton FP.Structural QuantizeIO
noncomputable section

def domain {n : Nat} (xs : LogicalView.Inputs n) : Prop :=
  0≤xs.maximum ∧ ∀ i, |xs.values i|≤xs.maximum

def formula {n : Nat} (xs : LogicalView.Inputs n) (i : Fin n) : Int :=
  if xs.maximum=0 then 0 else QuantizationReal.nearestEven (127*xs.values i/xs.maximum)

theorem judgment_iff (rest : Algebra ℝ) (p : Parameters) (trials : Nat → List Config) :
    (LogicalView.quantizeIO rest p trials).ImplementsOn domain formula ↔
    (io rest Real.arithmetic p trials).ImplementsOn (QuantizeIO.domain rest p trials)
      (target rest Real.arithmetic p trials (fun xs maximum i =>
        if maximum=0 then 0 else QuantizationReal.nearestEven (127*xs i/maximum))) := by
  unfold LogicalDomainIO.ImplementsOn LogicalDomainIO.lift
  apply Iff.of_eq
  congr 1
  · funext xs
    apply propext
    let values : TileIndex [p.n] → ℝ := xs (0 : Fin 2)
    let maximum : ℝ := xs (1 : Fin 2) ()
    change (0≤ maximum ∧ ∀ i : Fin p.n, |values (i,())|≤ maximum) ↔
      (0≤ maximum ∧ ∀ i : TileIndex [p.n], |values i|≤ maximum)
    constructor
    · rintro ⟨hn,h⟩; exact ⟨hn,fun j => h j.1⟩
    · rintro ⟨hn,h⟩; exact ⟨hn,fun i => h (i,())⟩
  · funext xs o j
    have ho : o=⟨0,by change 0<1; decide⟩ := Fin.eq_zero o
    subst o
    rfl
end
end QuantizeGlobalRepair.LogicalReader

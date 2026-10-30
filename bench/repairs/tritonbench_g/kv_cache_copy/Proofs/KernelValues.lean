import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelFP

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelFP
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 8000

theorem copyStep_readback {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s t : State α) (key : Bool) (q : Nat)
    (inj : Function.Injective (fun i : TileIndex [p.width] => (targetInt p s key q i).toNat))
    (i : TileIndex [p.width]) :
    (copyStep M p s q t).mem (outputRegion key) (targetInt p s key q i).toNat =
      .mk .real (M.fp32Load ((s.mem (inputRegion key) (sourceOffset p s.pids key q i)).read .real)) := by
  cases key with
  | «false» => exact State.scatter_readback _ _ _ _ i inj
  | «true» =>
    unfold copyStep
    rw [scatterOne_frame M p s _ Bool.false q _ _ (Or.inl (by decide))]
    exact State.scatter_readback _ _ _ _ i inj

/-- Readback from the complete split sequence; later writes cannot replace an
earlier K/V lane under the same explicit target injectivity as the Real theorem. -/
theorem copied_readback {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (n : Nat) (key : Bool)
    (q : Nat) (hq : q < n) (i : TileIndex [p.width])
    (unique : ∀ u v, u < n → v < n → ∀ j k : TileIndex [p.width],
      (targetInt p s key u j).toNat = (targetInt p s key v k).toNat → u = v ∧ j = k) :
    (copied M p s n).mem (outputRegion key) (targetInt p s key q i).toNat =
      .mk .real (M.fp32Load ((s.mem (inputRegion key) (sourceOffset p s.pids key q i)).read .real)) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases last : q = n
    · subst q
      exact copyStep_readback M p s _ key n
        (fun i j eq => (unique n n (by omega) (by omega) i j eq).2) i
    · have earlier : q < n := by omega
      rw [copied, copyStep_frame M p s _ n _ _ (by
        intro other
        by_cases same : key = other
        · subst other
          exact Or.inr (fun j eq => last ((unique n q (by omega) hq j i eq.symm).1.symm))
        · exact Or.inl (by cases key <;> cases other <;> simp_all [outputRegion]))]
      exact ih earlier (fun u v hu hv j k eq => unique u v (by omega) (by omega) j k eq)

#axiomsClean copyStep_readback
#axiomsClean copied_readback
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelFP

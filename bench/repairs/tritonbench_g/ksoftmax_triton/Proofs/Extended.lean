import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.ValueMath

namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- Actual typed forward store with negative infinity preserved. -/
theorem forward_extended_readback (Y X Mask : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0<D)
    (log : Bool) (mode : Forward.MaskMode) (causal : Bool)
    (s : State (WithBot ℝ)) (xs masks : Fin D → ℝ)
    (hx : ∀ j, Forward.readActive K (s.pids 1) causal j.val=Bool.true →
      (s.mem X (s.pids 0*xm+s.pids 1*xn+j.val)).read .real=some (xs j))
    (hm : ∀ j, mode.present=Bool.true → Forward.readActive K (s.pids 1) causal j.val=Bool.true →
      (s.mem Mask (Forward.maskAddress mode (s.pids 0) (s.pids 1) sm j.val)).read .real=some (masks j))
    (i : Fin D) (hi : i.val<K) :
    (Floating.forward_run Forward.TypedExecution.model Y X Mask ym yn xm xn sm K D hD
      log mode causal s).val.mem Y (s.pids 0*ym+s.pids 1*yn+i.val)=
      Cell.mk .real (Forward.extendedFormula K (s.pids 1) log causal mode xs masks i) := by
  rw [forward_value Forward.TypedExecution.model Y X Mask ym yn xm xn sm K D hD log mode causal s i hi]
  have hK : 0<K := by omega
  have result := forwardValue_real K (s.pids 1) hK hD log causal mode
    (fun j => (s.mem X (s.pids 0*xm+s.pids 1*xn+j.val)).read .real)
    (fun j => (s.mem Mask (Forward.maskAddress mode (s.pids 0) (s.pids 1) sm j.val)).read .real)
    xs masks hx hm i
  rw [←forward_formula K (s.pids 1) hK hD log causal mode xs masks i] at result
  exact result

#axiomsClean forward_extended_readback
end
end VeriTile.Bench.Repairs.Ksoftmax

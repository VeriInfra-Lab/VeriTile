import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeRawRecurrence
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeSourceExecution

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

def nativeRawValue {α : Type} [Inhabited α] (a : Arithmetic α) (plan : RotaryPlan) (fp8 : Bool)
    (p : Arguments) (s : State α) (epsilon theta : α) (batch row col : Nat) : α :=
  nativeReferenceValue a plan (parameters p epsilon theta) (rawRow a fp8 p s batch row)
    p.rotary row col

def nativeRawTile {α : Type} [Inhabited α] (a : Arithmetic α) (plan : RotaryPlan) (fp8 : Bool)
    (p : Arguments) (s : State α) (epsilon theta : α) : Values α .real [16,64] := fun i =>
  nativeRawValue a plan fp8 p s epsilon theta (s.pids 0)
    (programRow p s*16+i.1.val) (programCol p s*64+i.2.1.val)

theorem nativeTile_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (plan : RotaryPlan) (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (even : p.N%2=0) (row : Fin 16) (lane : Fin 64)
    (hm : programRow p s*16+row.val<p.M)
    (hn : programCol p s*64+lane.val<p.N) :
    nativeTileValues a plan fp8 p s epsilon theta (row,lane,PUnit.unit)=
      nativeRawTile a plan fp8 p s epsilon theta (row,lane,PUnit.unit) := by
  let moments := (nativeValuesAt a fp8 p s ((p.K+63)/64)).moments
  let accumulator := (nativeValuesAt a fp8 p s ((p.K+63)/64)).accumulator
  let normed := normalized a p epsilon moments accumulator
  let project := nativeReferenceProjection a.toScalar (parameters p epsilon theta)
    (rawRow a fp8 p s (s.pids 0) (programRow p s*16+row.val))
    (fun _ _ => a.plan) a.reduction
  have hnrm : ∀ (n : Fin 64),programCol p s*64+n.val<p.N →
      normed (row,n,PUnit.unit)=project (programCol p s*64+n.val) := by
    intro n hn
    simpa only [Nat.mod_eq_of_lt hm,Nat.mod_eq_of_lt hn] using
      native_normalized_reference a z fp8 p s epsilon theta row n
  change nativeEpilogueValues a plan p epsilon theta (programRow p s) (programCol p s)
    moments accumulator (row,lane,PUnit.unit)=_
  cases hr : p.rotary with
  | «false» =>
    simpa only [nativeEpilogueValues,nativeRawTile,nativeRawValue,nativeReferenceValue,nativeScalarPair,hr,if_false] using hnrm lane hn
  | «true» =>
    let pair : Fin 32 := ⟨lane.val/2,by have := lane.isLt; omega⟩
    have he : programCol p s*64+(evenLane pair).val<p.N := by
      simp only [evenLane,pair]
      omega
    have ho : programCol p s*64+(oddLane pair).val<p.N := by
      simp only [oddLane,pair]
      omega
    have epair : programCol p s*64+(evenLane pair).val=
        (programCol p s*64+lane.val)-(programCol p s*64+lane.val)%2 := by
      simp only [evenLane,pair]
      omega
    have opair : programCol p s*64+(oddLane pair).val=
        (programCol p s*64+lane.val)-(programCol p s*64+lane.val)%2+1 := by
      simp only [oddLane,pair]
      omega
    have parity : (programCol p s*64+lane.val)%2=lane.val%2 := by omega
    simp only [nativeEpilogueValues,hr,if_true]
    change (if lane.val%2=0 then
      (nativeRotatedPair a plan p theta (programRow p s) (programCol p s) normed row pair).1 else
      (nativeRotatedPair a plan p theta (programRow p s) (programCol p s) normed row pair).2)=_
    simp only [nativeRotatedPair,hnrm (evenLane pair) he,hnrm (oddLane pair) ho,
      epair,opair,nativeRawTile,nativeRawValue,nativeReferenceValue,nativeScalarPair,hr,if_true,parameters,
      pairAngle,pairColumn,parity]
    have phasePair : programCol p s*64+2*pair.val=
        (programCol p s*64+lane.val)-(programCol p s*64+lane.val)%2 := epair
    simp only [phasePair,parity]
    rfl

#axiomsClean nativeTile_reference
#specNonCircular nativeRawValue avoiding [nativeTileValues,nativeLoopValues,run,value,accumulation,moment]
#specNonCircular nativeRawTile avoiding [nativeTileValues,nativeLoopValues,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain

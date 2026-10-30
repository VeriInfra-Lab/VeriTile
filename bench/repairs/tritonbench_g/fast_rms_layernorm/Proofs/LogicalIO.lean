import bench.repairs.tritonbench_g.fast_rms_layernorm.IO
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealPublicIO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.LogicalIO
open VeriTile Triton
open scoped VeriTile.Triton.LogicalIO VeriTile.Triton.LogicalDomainIO
noncomputable section

@[simp] theorem x_row (h : Call) (raw : Ports.Values ℝ h) : (inputs h raw).x = Reference.row h raw := by
  funext row col
  change raw 0 (finProdFinEquiv (row,col)) = raw 0 _
  congr 1
  apply Fin.ext
  simp [finProdFinEquiv,Nat.mul_comm,Nat.add_comm]
@[simp] theorem upstream_row (h : Call) (raw : Ports.Values ℝ h) : (inputs h raw).upstream = Reference.upstream h raw := by
  funext row col
  change raw 3 (finProdFinEquiv (row,col)) = raw 3 _
  congr 1
  apply Fin.ext
  simp [finProdFinEquiv,Nat.mul_comm,Nat.add_comm]

@[simp] theorem weight_eq (h : Call) (raw : Ports.Values ℝ h) : (inputs h raw).weight = raw 1 := rfl
@[simp] theorem saved_eq (h : Call) (raw : Ports.Values ℝ h) : (inputs h raw).savedR = raw 2 := rfl
@[simp] theorem initial_eq (h : Call) (raw : Ports.Values ℝ h) : (inputs h raw).initialDW = raw 4 := rfl

/-- Full prepared schedule, typed cell binding and frame are fixed in the index. -/
theorem forward_iff (h : Call) (f : Inputs h → ForwardOutputs h) :
    (forwardView h ⊨ f) ↔ (RealIO.program h Bool.false).Implements
      (fun raw => Fin.cases (fun j => (f (inputs h raw)).values j.divNat j.modNat)
        (Fin.cases (f (inputs h raw)).savedR (fun i => i.elim0))) := Iff.rfl

theorem backward_iff (h : Call) (domain : Inputs h → Prop) (f : Inputs h → BackwardOutputs h) :
    (backwardView h ⊨{domain} f) ↔ (RealIO.program h Bool.true).ImplementsOn (fun raw => domain (inputs h raw))
      (fun raw => Fin.cases (fun j => (f (inputs h raw)).inputGradient j.divNat j.modNat)
        (Fin.cases (f (inputs h raw)).weightGradient (fun i => i.elim0))) := Iff.rfl
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.LogicalIO

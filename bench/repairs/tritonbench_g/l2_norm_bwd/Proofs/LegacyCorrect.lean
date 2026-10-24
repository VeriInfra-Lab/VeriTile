import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Binding
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Certified
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Real

namespace VeriTile.Bench.Repairs.L2NormBwd
open VeriTile.Triton Physical Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

def realNegation : Negation ℝ := ⟨fun x => -x,neg_neg⟩

def realResult {M β : Type} (ops : Ops M ℝ β) (call : Call) (eps : ℝ) (initial : World M) (i : Nat) : ℝ :=
  Math.derivative call.cols eps
    (fun j => logical ops realNegation initial.memory call.x ((i/call.cols)*call.cols+j))
    (fun j => logical ops realNegation initial.memory call.dy ((i/call.cols)*call.cols+j)) (i%call.cols)

/-- Whole public wrapper in the exact-Real interpretation. The target is the
independently written logical derivative, including the original return shape.
Both successful and failing actual host paths preserve every caller byte. -/
theorem source_real {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops realNegation rt) (grids : Machine.GridContract rt) (call : Call) (eps : ℝ) (initial : World M)
    (domain : InputDomain call initial) :
    Ensures (fun output after =>
      WholeSpec ops realNegation (sourceReduction Math.algebra) rt call eps initial output after ∧
      Nonempty (StructuralCertificate ops Math.algebra rt call eps initial output after) ∧
      ∀ i,i<call.x.numel → logical ops realNegation after.memory output i=realResult ops call eps initial i)
      (fun _ after => Frame ops call.caller initial after)
      (Host.run ops (sourceReduction Math.algebra) rt call eps initial) := by
  have spec := whole_spec ops realNegation (sourceReduction Math.algebra) rt lib call eps initial domain
  cases outcome : Host.run ops (sourceReduction Math.algebra) rt call eps initial with
  | error reason after => rw [outcome] at spec;exact spec
  | ok output after =>
    rw [outcome] at spec
    change WholeSpec ops realNegation (sourceReduction Math.algebra) rt call eps initial output after at spec
    refine ⟨spec,structural_certificate ops realNegation Math.algebra rt lib grids call eps initial after output domain outcome,?_⟩
    intro i hi
    rw [spec.values i hi]
    unfold expected sourceReduction realResult
    dsimp only
    apply Math.source_derivative
    exact spec.block

/-- Python admission and the public input premise are equivalent. -/
theorem python_checks_iff (x dy : ArgumentMetadata) :
    checkInputs x dy=.ok () ↔ AdmittedArguments x dy := checkInputs_iff x dy

/-- Whole checked direct-view wrapper, retaining the independent Real formula,
returned allocation, structural source/machine execution and all caller bytes. -/
theorem whole_real {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops realNegation rt) (grids : Machine.GridContract rt)
    (x dy : TensorMetadata) (input gradient : View) (eps : ℝ) (initial : World M)
    (h : InputContract x dy) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : initial.live input.storage) (ld : initial.live gradient.storage) :
    let call : Call := ⟨input,gradient⟩
    match checkedRun ops (sourceReduction Math.algebra) rt x dy input gradient eps initial with
    | .error _ => True
    | .ok result => Ensures (fun output after =>
        WholeSpec ops realNegation (sourceReduction Math.algebra) rt call eps initial output after ∧
        Nonempty (StructuralCertificate ops Math.algebra rt call eps initial output after) ∧
        Nonempty (DirectPath ops (sourceReduction Math.algebra) rt call eps initial output after) ∧
        ∀ i,i< input.numel → logical ops realNegation after.memory output i=realResult ops call eps initial i)
        (fun _ after => Frame ops call.caller initial after) result := by
  dsimp only
  rw [checked_run _ _ _ _ _ _ _ _ _ h]
  have domain:=checked_domain x dy input gradient initial bx bd lx ld
  have spec:=source_real ops rt lib grids ⟨input,gradient⟩ eps initial domain
  cases outcome : Host.run ops (sourceReduction Math.algebra) rt ⟨input,gradient⟩ eps initial with
  | error reason after => rw [outcome] at spec;exact spec
  | ok output after =>
    rw [outcome] at spec
    exact ⟨spec.1,spec.2.1,direct_path ops realNegation (sourceReduction Math.algebra) rt lib
      ⟨input,gradient⟩ eps initial after output domain outcome,spec.2.2⟩

#axiomsClean python_checks_iff
#axiomsClean whole_real

#specNonCircular realResult avoiding [Host.run,Tile.scalar,source,Math.algebra]
#axiomsClean source_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd

import bench.repairs.tritonbench_g.rope_embedding.Proofs.KernelBinding
namespace VeriTile.Bench.Repairs.RopeEmbedding.RealBinding
open VeriTile.Triton FP.Structural
open VeriTile.Torch
open KernelBinding (Inputs)
open Numerics (Word)
noncomputable section
set_option maxHeartbeats 2000000

/-- Interpret the actual FP32 input reads over Real. The interpretation is an
explicit input-model parameter; this does not assert exact Real FP32 execution.
Only Q/C/S cells are replaced. All other memory and registers come from scratch. -/
def state (read : Nat → Word) (interpret : Word → ℝ) (inputs : Inputs)
    (scratch : State ℝ) : State ℝ :=
  { scratch with
    mem := fun region o =>
      if region = "Q" ∨ region = "C" ∨ region = "S" then
        Cell.mk .real (interpret (KernelBinding.memory read inputs
          (fun _ _ => FP.Value.zero .fp32) region o))
      else
        scratch.mem region o }

/-- Shared Tensor laws bind the interpreted physical Q read to its logical
input element; no kernel output or input-value equality is assumed. -/
theorem input (read : Nat → Word) (interpret : Word → ℝ) (inputs : Inputs)
    (r : Request) (scratch : State ℝ)
    (accepted : (KernelBinding.metadata inputs r).check = Bool.true)
    (row head col : Nat) (hr : row < r.rows) (hh : head < r.heads) (hc : col < r.dimension) :
    ((state read interpret inputs scratch).mem "Q" (r.offset row head col)).read .real =
      interpret (Torch.FP32.logicalValue read inputs.q inputs.qLaws (r.offset row head col)) := by
  change interpret (KernelBinding.memory read inputs (fun _ _ => FP.Value.zero .fp32)
    "Q" (r.offset row head col)) = _
  exact congrArg interpret (KernelBinding.input_binding read inputs r _ accepted row head col hr hh hc)

/-- Both coefficient tables use the same actual pointer/stride interpretation
as the FP endpoint and the existing Python checks. -/
theorem coefficients (read : Nat → Word) (interpret : Word → ℝ) (inputs : Inputs)
    (r : Request) (scratch : State ℝ)
    (accepted : (KernelBinding.metadata inputs r).check = Bool.true)
    (row lane : Nat) (hr : row < r.rows) (hl : lane < r.half) :
    ((state read interpret inputs scratch).mem "C" (row % r.sequence * r.half + lane)).read .real =
      interpret (Torch.FP32.logicalValue read inputs.cosine inputs.cosineLaws
        (row % r.sequence * r.half + lane)) ∧
    ((state read interpret inputs scratch).mem "S" (row % r.sequence * r.half + lane)).read .real =
      interpret (Torch.FP32.logicalValue read inputs.sine inputs.sineLaws
        (row % r.sequence * r.half + lane)) := by
  obtain ⟨hc, hs⟩ := KernelBinding.coefficient_binding read inputs r
    (fun _ _ => FP.Value.zero .fp32) accepted row lane hr hl
  exact ⟨congrArg interpret hc, congrArg interpret hs⟩

/-- Substitute derived logical input reads into the independent mathematical
formula, including its unchanged odd tail. -/
theorem result (read : Nat → Word) (interpret : Word → ℝ) (inputs : Inputs)
    (r : Request) (back : Bool) (scratch : State ℝ)
    (accepted : (KernelBinding.metadata inputs r).check = Bool.true)
    (row head col : Nat) (hr : row < r.rows) (hh : head < r.heads) (hc : col < r.dimension) :
    Spec.result r back
      (fun o => ((state read interpret inputs scratch).mem "Q" o).read .real)
      (fun o => ((state read interpret inputs scratch).mem "C" o).read .real)
      (fun o => ((state read interpret inputs scratch).mem "S" o).read .real) row head col =
    Spec.result r back
      (fun o => interpret (Torch.FP32.logicalValue read inputs.q inputs.qLaws o))
      (fun o => interpret (Torch.FP32.logicalValue read inputs.cosine inputs.cosineLaws o))
      (fun o => interpret (Torch.FP32.logicalValue read inputs.sine inputs.sineLaws o)) row head col := by
  unfold Spec.result
  split
  · rename_i active
    let lane := if col < r.half then col else col - r.half
    have hl : lane < r.half := by dsimp [lane]; split <;> omega
    have dimension : 2 * r.half ≤ r.dimension := by unfold Request.half; omega
    have q1 := input read interpret inputs r scratch accepted row head lane hr hh (by omega)
    have q2 := input read interpret inputs r scratch accepted row head (lane + r.half) hr hh (by omega)
    obtain ⟨cosine, sine⟩ := coefficients read interpret inputs r scratch accepted row lane hr hl
    dsimp only
    rw [q1, q2, cosine, sine]
  · exact input read interpret inputs r scratch accepted row head col hr hh hc

#axiomsClean input
#axiomsClean coefficients
#axiomsClean result
end
end VeriTile.Bench.Repairs.RopeEmbedding.RealBinding

import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.SetupCalls
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostBlock

/-! Exact return-before-scale-validation prefix. In particular D=0 imposes
no scale-length or launch/native precondition. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.HostEmpty
open Verified.HostSetup HostShape
noncomputable section

def prepareThenEmpty (ok : Call → Bool) (output : OutputDescriptor) : Except Call (Option OutputDescriptor) :=
  match calls ok beforeEmpty with
  | .error c => .error c
  | .ok _ => .ok (if output.batches*output.heads*output.context*output.headDim=0 then some output else none)

/-- The exact fresh BF16 descriptor is returned after the ordinary successful
preparation/allocation prefix; caller-owned memory is preserved. No scale
length, element input, compiler or launch premise is present. -/
theorem empty_return {Cell : Type} (capacity : Nat) (caller prepared : Fin capacity → Cell)
    (owned fresh : Fin capacity → Prop) (output : OutputDescriptor) (ok : Call → Bool)
    (hprefix : calls ok beforeEmpty=.ok ()) (hdtype : output.dtype=VeriTile.Triton.TileDType.bf16)
    (hempty : output.batches*output.heads*output.context*output.headDim=0)
    (hframe : ∀ cell, ¬fresh cell → prepared cell=caller cell)
    (hfresh : ∀ cell, owned cell → ¬fresh cell) :
    prepareThenEmpty ok output=.ok (some output) ∧ output.dtype=VeriTile.Triton.TileDType.bf16 ∧
    IsEmpty (Fin output.batches × Fin output.heads × Fin output.context × Fin output.headDim) ∧
    (∀ cell, owned cell → prepared cell=caller cell) := by
  refine ⟨by simp [prepareThenEmpty,hprefix,hempty],hdtype,⟨?_⟩,fun cell hc => hframe cell (hfresh cell hc)⟩
  intro c
  have hB := c.1.isLt
  have hH := c.2.1.isLt
  have hN := c.2.2.1.isLt
  have hD := c.2.2.2.isLt
  have hp := Nat.mul_pos (Nat.mul_pos (Nat.mul_pos (by omega : 0<output.batches)
    (by omega : 0<output.heads)) (by omega : 0<output.context)) (by omega : 0<output.headDim)
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.HostEmpty

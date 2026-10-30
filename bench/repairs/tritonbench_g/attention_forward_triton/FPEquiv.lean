import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.HoistedIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Full-grid equivalence of the repaired FP32 source and an invariant-mask
hoist. The candidate computes the head mask once before the loop and reuses it
at every Q/K/V load and output store. Floating arithmetic is unchanged. -/
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.FPEquiv
open VeriTile Triton FP.Structural Source Whole Spec
noncomputable section

def original (I : Native.Instructions) (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.original (AttentionIO.nativeIO I p work) (fun _=>True)

def candidate (I : Native.Instructions) (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.ofVariant (io:=AttentionIO.nativeIO I p work)
    {schedule:=(HoistedIO.nativeIO I p work).schedule,
     observe:=(HoistedIO.nativeIO I p work).observe} (fun _=>True)

/-- Every legal full-grid order and all FP32 input words are retained. Both
actual sources execute safely, every typed FP16 output agrees under the same
native observer, and both full exterior frames hold. There is no numerical
rewrite or finite-word premise. -/
specification attention_head_mask_equiv (I : Native.Instructions)
    (p : Parameters) (sep : Separated p) (injective : RawKernel.OutputInjective p)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    original I p work ≡[[]] candidate I p work := by
  apply ProgramIO.Implementation.of_equivalent
  exact ProgramIO.EquivalentOn.of_implementsOn
    (AttentionIO.native_implements I p sep injective work legal)
    (HoistedIO.native_implements I p sep injective work legal)

#axiomsClean attention_head_mask_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.FPEquiv

import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedGeometry
import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedExecution
import VeriTile.Triton.Float.ProfiledSoundness

/-! Ordinary disjoint-program opaque launch. Reference cells come from the
unchanged division source on the initial caller memory. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

structure GridRun {α : Type} [Inhabited α] (A : Algebra α) (p : Config) (BD : Nat)
    (r : Bool) (initial final : State α) where
  programs : ProgramIndex p → State α
  executed : ∀ idx : ProgramIndex p, exec (engine A) (source p BD r) (atProgram p initial idx)=some (programs idx)
  framed : ∀ idx : ProgramIndex p, Frame p BD idx.1.val idx.2.val (atProgram p initial idx) (programs idx)
  disjoint : ∀ (i j : ProgramIndex p) region offset, Written p BD i.1.val i.2.val region offset →
    Written p BD j.1.val j.2.val region offset → i=j
  readback : ∀ (idx : ProgramIndex p) region offset, Written p BD idx.1.val idx.2.val region offset →
    final.mem region offset=(programs idx).mem region offset
  frame : ∀ region offset, ¬Window p region offset → final.mem region offset=initial.mem region offset

def referenceState {α : Type} [Inhabited α] (A : Algebra α) (p : Config) (BD : Nat)
    (support : DotSupport A BD) (initial : State α) (idx : ProgramIndex p) : State α :=
  Classical.choose (success A p BD support Bool.false (atProgram p initial idx))

theorem reference_exec {α : Type} [Inhabited α] (A : Algebra α) (p : Config) (BD : Nat)
    (support : DotSupport A BD) (initial : State α) (idx : ProgramIndex p) :
    exec (engine A) (source p BD Bool.false) (atProgram p initial idx)=
      some (referenceState A p BD support initial idx) :=
  Classical.choose_spec (success A p BD support Bool.false (atProgram p initial idx))

def reference {α : Type} [Inhabited α] (A : Algebra α) (p : Config) (BD : Nat)
    (support : DotSupport A BD) (initial : State α) (c : Coordinate p) : Cell α :=
  (referenceState A p BD support initial (owner p c)).mem p.Out (logicalAddress p c)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

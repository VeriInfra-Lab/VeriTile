import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedGeometry
import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedFP

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

/-- Both actual source grids successfully produce the same observable final
state, with exact program ownership and the full complement preserved. -/
theorem grid_equivalence {α : Type} [Inhabited α] (R : FPEquivProofs.Rules)
    (A : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions A D)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (initial : State α)
    (hD : p.headDim≤BD) (hn : NonOverlapping p)
    (hd : ∀ idx : ProgramIndex p, ∀ site∈(program p BD Bool.false).domain,
      site.Holds A D (atProgram p initial idx)) :
    ∃ final,
      Nonempty (GridRun A p BD Bool.false initial final) ∧
      Nonempty (GridRun A p BD Bool.true initial final) ∧
      (∀ c : Coordinate p, final.mem p.Out (logicalAddress p c)=reference A p BD support initial c) ∧
      (∀ region offset, ¬Window p region offset → final.mem region offset=initial.mem region offset) := by
  have hprograms : ∀ idx : ProgramIndex p, ∃ t,
      exec (engine A) (source p BD Bool.false) (atProgram p initial idx)=some t ∧
      exec (engine A) (source p BD Bool.true) (atProgram p initial idx)=some t ∧
      Frame p BD idx.1.val idx.2.val (atProgram p initial idx) t := by
    intro idx
    obtain ⟨t,ht,hr⟩ := successful_equivalence R A D hM p BD support (atProgram p initial idx) (hd idx)
    obtain ⟨u,hu,hf⟩ := executes_framed A p BD support Bool.false (atProgram p initial idx)
    have he : u=t := Option.some.inj (hu.symm.trans ht)
    subst u
    exact ⟨t,ht,hr,hf⟩
  choose programs hexec hrecip hframe using hprograms
  let final : State α := {initial with mem := fun region offset =>
    if h : (∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset) then
      ((programs h.choose).mem region offset) else (initial.mem region offset)}
  have hread : ∀ (idx : ProgramIndex p) region offset, Written p BD idx.1.val idx.2.val region offset →
      final.mem region offset=(programs idx).mem region offset := by
    intro idx region offset hw
    have hex : ∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset := ⟨idx,hw⟩
    change (if h : (∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset) then
      ((programs h.choose).mem region offset) else (initial.mem region offset))=_
    rw [dif_pos hex,disjoint p BD hn hex.choose idx region offset hex.choose_spec hw]
  have hf : ∀ region offset, ¬Window p region offset → final.mem region offset=initial.mem region offset := by
    intro region offset hnot
    change (if h : (∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset) then
      ((programs h.choose).mem region offset) else (initial.mem region offset))=_
    apply dif_neg
    rintro ⟨idx,hw⟩
    exact hnot (written_within p BD idx region offset hw)
  refine ⟨final,⟨⟨programs,hexec,hframe,disjoint p BD hn,hread,hf⟩⟩,
    ⟨⟨programs,hrecip,hframe,disjoint p BD hn,hread,hf⟩⟩,?_,hf⟩
  intro c
  rw [hread (owner p c) _ _ (covered p BD hD c)]
  have he := Option.some.inj ((hexec (owner p c)).symm.trans (reference_exec A p BD support initial (owner p c)))
  rw [he]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.DirectBinding

namespace VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost
open Physical

def activeRole (backward : Bool) (r : Role) : Prop := r ≠ .third ∨ backward = Bool.true
instance (backward : Bool) (r : Role) : Decidable (activeRole backward r) :=
  inferInstanceAs (Decidable (r ≠ .third ∨ backward = Bool.true))

/-- The source's input list has A/B in forward and A/B/DC in backward.
Empty-like output properties are allocator laws, not user assertions. -/
structure InputAdmitted (p : Prepared backward) : Prop where
  payloads : ∀ r, activeRole backward r → payload (p.tensor r) = Bool.true
  counts : ∀ r, activeRole backward r → (p.tensor r).metadata.numel = p.rows*p.cols
  forwardShape : backward = Bool.false → (p.tensor .b).metadata.shape = p.originalShape
  sameDevice : ∀ r, activeRole backward r → (p.tensor r).metadata.device = (p.tensor .a).metadata.device
  positive : 0 < p.cols
  grid32 : (p.rows*p.cols+127)/128 < 2^31
  padded64 : p.rows*p.cols+127 < 2^63
  alignment : 0 < p.rows*p.cols → ∀ r, activeRole backward r → (p.tensor r).metadata.dataPtr.toNat % 4 = 0
  separate : 0 < p.rows*p.cols → ∀ r s, activeRole backward r → activeRole backward s →
    r ≠ s → spanSeparate (p.tensor r) (p.tensor s)

def inputCheck (p : Prepared backward) : Bool :=
  decide (∀ r, activeRole backward r → payload (p.tensor r) = Bool.true) &&
  decide (∀ r, activeRole backward r → (p.tensor r).metadata.numel = p.rows*p.cols) &&
  decide (backward = Bool.false → (p.tensor .b).metadata.shape = p.originalShape) &&
  decide (∀ r, activeRole backward r → (p.tensor r).metadata.device = (p.tensor .a).metadata.device) &&
  decide (0 < p.cols) && decide ((p.rows*p.cols+127)/128 < 2^31) && decide (p.rows*p.cols+127 < 2^63) &&
  decide (0 < p.rows*p.cols → ∀ r, activeRole backward r → (p.tensor r).metadata.dataPtr.toNat % 4 = 0) &&
  decide (0 < p.rows*p.cols → ∀ r s, activeRole backward r → activeRole backward s →
    r ≠ s → spanSeparate (p.tensor r) (p.tensor s))

theorem inputCheck_iff (p : Prepared backward) : inputCheck p = Bool.true ↔ InputAdmitted p := by
  simp only [inputCheck,Bool.and_eq_true,decide_eq_true_eq]
  exact ⟨fun ⟨⟨⟨⟨⟨⟨⟨⟨a,b⟩,c⟩,d⟩,e⟩,f⟩,g⟩,i⟩,j⟩ => ⟨a,b,c,d,e,f,g,i,j⟩,
    fun h => ⟨⟨⟨⟨⟨⟨⟨⟨h.payloads,h.counts⟩,h.forwardShape⟩,h.sameDevice⟩,h.positive⟩,h.grid32⟩,h.padded64⟩,h.alignment⟩,h.separate⟩⟩

theorem inactive (backward : Bool) (r : Role) (off : ¬activeRole backward r) : r = .third ∧ backward = Bool.false := by
  cases backward <;> cases r <;> simp_all [activeRole]

theorem input_admitted (p : Prepared backward) (h : InputAdmitted p) : Admitted p := by
  refine ⟨?_,?_,h.forwardShape,?_,h.positive,h.grid32,h.padded64,?_,?_⟩
  · intro r
    by_cases on : activeRole backward r
    · exact h.payloads r on
    · obtain ⟨rfl,off⟩ := inactive backward r on
      exact p.allocatedPayload off
  · intro r
    by_cases on : activeRole backward r
    · exact h.counts r on
    · obtain ⟨rfl,off⟩ := inactive backward r on
      exact p.allocatedCount off
  · intro r
    by_cases on : activeRole backward r
    · exact h.sameDevice r on
    · obtain ⟨rfl,off⟩ := inactive backward r on
      exact p.allocatedDevice off
  · intro nonempty r
    by_cases on : activeRole backward r
    · exact h.alignment nonempty r on
    · obtain ⟨rfl,off⟩ := inactive backward r on
      exact p.allocatedAlignment off nonempty
  · intro nonempty r s ne
    by_cases onr : activeRole backward r
    · by_cases ons : activeRole backward s
      · exact h.separate nonempty r s onr ons ne
      · obtain ⟨rfl,off⟩ := inactive backward s ons
        have separate := p.allocatedSeparate off nonempty
        cases r
        · exact separate.1
        · exact separate.2
        · exact False.elim (ne rfl)
    · obtain ⟨rfl,off⟩ := inactive backward r onr
      have separate := p.allocatedSeparate off nonempty
      cases s
      · exact separate.1.symm
      · exact separate.2.symm
      · exact False.elim (ne rfl)

theorem inputCheck_eq (p : Prepared backward) : inputCheck p = check p := by
  apply Bool.eq_iff_iff.mpr
  rw [inputCheck_iff,check_iff]
  exact ⟨input_admitted p,fun h => ⟨fun r _ => h.payloads r,fun r _ => h.counts r,
    h.forwardShape,fun r _ => h.sameDevice r,h.positive,h.grid32,h.padded64,
    fun hne r _ => h.alignment hne r,fun hne r s _ _ ne => h.separate hne r s ne⟩⟩

theorem inputCheck_admitted_iff (p : Prepared backward) : inputCheck p = Bool.true ↔ Admitted p := by
  rw [inputCheck_eq,check_iff]

#axiomsClean inputCheck_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost

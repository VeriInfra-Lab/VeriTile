import bench.tritonbench_g.kldiv_ops.Proofs.Library

namespace VeriTile.Bench.Repairs.KldivOps.Host
noncomputable section

inductive ReverseProject : List Nat→List Nat→Prop
  | nil (out : List Nat) : ReverseProject [] out
  | cons {d o : Nat} {ds os : List Nat} : (d=1 ∨ d=o) → ReverseProject ds os → ReverseProject (d::ds) (o::os)

theorem ReverseProject.refl (shape : List Nat) : ReverseProject shape shape := by
  induction shape with
  | nil => exact .nil _
  | cons d ds ih => exact .cons (Or.inr rfl) ih

theorem broadcastReverse_project (a b out : List Nat) (h : broadcastReverse a b=some out) :
    ReverseProject a out ∧ ReverseProject b out := by
  induction a generalizing b out with
  | nil =>
    simp only [broadcastReverse,Option.some.injEq] at h
    subst out
    exact ⟨.nil _,ReverseProject.refl _⟩
  | cons d ds ih =>
    cases b with
    | nil =>
      simp only [broadcastReverse,Option.some.injEq] at h
      subst out
      exact ⟨ReverseProject.refl _,.nil _⟩
    | cons o os =>
      cases hr : broadcastReverse ds os with
      | none => simp [broadcastReverse,hr] at h
      | some rest =>
        have tail := ih os rest hr
        by_cases eq : d=o
        · simp [broadcastReverse,hr,eq] at h
          subst out
          exact ⟨.cons (Or.inr eq) tail.1,.cons (Or.inr rfl) tail.2⟩
        · by_cases one : d=1
          · simp only [broadcastReverse,hr,Option.bind_some,if_neg eq,if_pos one,Option.some.injEq] at h
            simp at h
            subst out
            exact ⟨.cons (Or.inl one) tail.1,.cons (Or.inr rfl) tail.2⟩
          · by_cases other : o=1
            · simp only [broadcastReverse,hr,Option.bind_some,if_neg eq,if_neg one,if_pos other,Option.some.injEq] at h
              simp at h
              subst out
              exact ⟨.cons (Or.inr rfl) tail.1,.cons (Or.inl other) tail.2⟩
            · simp [broadcastReverse,hr,eq,one,other] at h

theorem ReverseProject.index_bound {input out : List Nat} (project : ReverseProject input out)
    (i : Nat) (hi : i<out.prod) : broadcastIndexReverse out input i < input.prod := by
  induction project generalizing i with
  | nil out => cases out <;> simp [broadcastIndexReverse]
  | @cons d o ds os dim rest ih =>
    have ho : 0<o := by
      by_contra h
      have hz : o=0 := by omega
      simp [hz] at hi
    have hq : i/o<os.prod := (Nat.div_lt_iff_lt_mul ho).mpr (by simpa [List.prod_cons,Nat.mul_comm] using hi)
    have bound := ih (i/o) hq
    simp only [broadcastIndexReverse,List.prod_cons]
    rcases dim with one | same
    · simpa [one] using bound
    · subst d
      by_cases one : o=1
      · simpa [one] using bound
      · have hm := Nat.mod_lt i ho
        simp only [one,if_false]
        nlinarith

/-- Output linear coordinates always project to valid logical coordinates in
both input tensors, including scalar, singleton, higher-rank and empty shapes. -/
theorem broadcast_indices (a b out : List Nat) (shape : broadcastShape a b=some out)
    (i : Nat) (hi : i<out.prod) : broadcastIndex out a i<a.prod ∧ broadcastIndex out b i<b.prod := by
  cases h : broadcastReverse a.reverse b.reverse with
  | none => simp [broadcastShape,h] at shape
  | some reversed =>
    have eq : reversed.reverse=out := by simpa [broadcastShape,h] using shape
    have project := broadcastReverse_project a.reverse b.reverse reversed h
    subst out
    have bound : i<reversed.prod := by simpa using hi
    constructor
    · simpa [broadcastIndex] using project.1.index_bound i bound
    · simpa [broadcastIndex] using project.2.index_bound i bound

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host

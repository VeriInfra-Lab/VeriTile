import bench.tritonbench_g.isfinite_kernel.Proofs.SourceRun
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryLaws

/-! Independent pointwise bit classification and full tagged memory frame.
The pure memory proof follows source tile order but permits repeated tile
IDs. Disjoint input/output regions preserve input bits across iterations;
positive output stride makes the logical output index unambiguous. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural
noncomputable section

def TileCovers (p : Parameters) (base x : Nat) : Prop :=
  ∃ i : Fin p.block, base + i.val = x ∧ x < p.size

def IterCovers (p : Parameters) (pid numCtas : Nat) : Nat → Nat → Nat → Prop
  | 0, _, _ => False
  | n + 1, cur, x =>
      TileCovers p ((pid + cur * numCtas) * p.block) x ∨
      IterCovers p pid numCtas n (cur + 1) x

def SourceCovers (p : Parameters) (pid numCtas x : Nat) : Prop :=
  if p.oneTile then TileCovers p (pid * p.block) x
  else IterCovers p pid numCtas p.iterations 0 x

theorem tileMemory_cover {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (base : Nat) (s : State α)
    (hstride : 0 < p.outputStride) (x : Nat) (hx : TileCovers p base x) :
    tileMemory classify p base s.mem p.output (x * p.outputStride) =
      .mk .bool (classify ((s.mem p.input (x * p.inputStride)).read .real)) := by
  obtain ⟨i, rfl, hi⟩ := hx
  exact tileMemory_readback classify p base s i hi hstride

theorem tileMemory_frame_cover {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (base : Nat) (s : State α)
    (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ x, TileCovers p base x → o ≠ x * p.outputStride) :
    tileMemory classify p base s.mem r o = s.mem r o := by
  apply tileMemory_frame
  rcases hmiss with hr | ho
  · exact Or.inl hr
  · exact Or.inr (fun i hi => ho (base + i.val) ⟨i, rfl, hi⟩)

theorem iterateMemory_contract {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (pid numCtas : Nat)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (n cur : Nat) (s : State α) :
    (∀ x, IterCovers p pid numCtas n cur x →
      iterateMemory
        (fun j => tileMemory classify p ((pid + j * numCtas) * p.block))
        n cur s.mem p.output (x * p.outputStride) =
      .mk .bool (classify ((s.mem p.input (x * p.inputStride)).read .real))) ∧
    (∀ r o, (r ≠ p.output ∨ ∀ x, IterCovers p pid numCtas n cur x →
        o ≠ x * p.outputStride) →
      iterateMemory
        (fun j => tileMemory classify p ((pid + j * numCtas) * p.block))
        n cur s.mem r o = s.mem r o) := by
  induction n generalizing cur s with
  | zero =>
      constructor
      · intro x hx
        exact False.elim hx
      · intro r o _
        rfl
  | succ n ih =>
      let base := (pid + cur * numCtas) * p.block
      let mid : State α := { s with mem := tileMemory classify p base s.mem }
      obtain ⟨hread, hframe⟩ := ih (cur + 1) mid
      constructor
      · intro x hx
        change iterateMemory
          (fun j => tileMemory classify p ((pid + j * numCtas) * p.block))
          n (cur + 1) mid.mem p.output (x * p.outputStride) = _
        by_cases hlater : IterCovers p pid numCtas n (cur + 1) x
        · rw [hread x hlater]
          change Cell.mk .bool (classify ((tileMemory classify p base s.mem p.input
            (x * p.inputStride)).read .real)) = _
          rw [tileMemory_input classify p base s hne]
        · have hfirst : TileCovers p base x := Or.resolve_right hx hlater
          rw [hframe p.output (x * p.outputStride) (Or.inr (by
            intro y hy heq
            have hxy : x = y := Nat.mul_right_cancel hstride heq
            exact hlater (hxy.symm ▸ hy)))]
          exact tileMemory_cover classify p base s hstride x hfirst
      · intro r o hmiss
        change iterateMemory
          (fun j => tileMemory classify p ((pid + j * numCtas) * p.block))
          n (cur + 1) mid.mem r o = s.mem r o
        have hrest : r ≠ p.output ∨ ∀ x,
            IterCovers p pid numCtas n (cur + 1) x → o ≠ x * p.outputStride := by
          rcases hmiss with hr | ho
          · exact Or.inl hr
          · exact Or.inr (fun x hx => ho x (Or.inr hx))
        rw [hframe r o hrest]
        apply tileMemory_frame_cover classify p base s r o
        rcases hmiss with hr | ho
        · exact Or.inl hr
        · exact Or.inr (fun x hx => ho x (Or.inl hx))

theorem sourceMemory_contract {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (pid numCtas : Nat)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride) (s : State α) :
    (∀ x, SourceCovers p pid numCtas x →
      sourceMemory classify p pid numCtas s.mem p.output (x * p.outputStride) =
      .mk .bool (classify ((s.mem p.input (x * p.inputStride)).read .real))) ∧
    (∀ r o, (r ≠ p.output ∨ ∀ x, SourceCovers p pid numCtas x →
        o ≠ x * p.outputStride) →
      sourceMemory classify p pid numCtas s.mem r o = s.mem r o) := by
  cases hb : p.oneTile with
  | «true» =>
      simp only [SourceCovers, sourceMemory, hb, ↓reduceIte]
      exact ⟨tileMemory_cover classify p (pid * p.block) s hstride,
        tileMemory_frame_cover classify p (pid * p.block) s⟩
  | «false» =>
      simp only [SourceCovers, sourceMemory, hb, Bool.false_eq_true, if_false]
      exact iterateMemory_contract classify p pid numCtas hne hstride p.iterations 0 s

theorem iterCovers_iff (p : Parameters) (pid numCtas n cur x : Nat) :
    IterCovers p pid numCtas n cur x ↔
      ∃ j, cur ≤ j ∧ j < cur + n ∧
        TileCovers p ((pid + j * numCtas) * p.block) x := by
  induction n generalizing cur with
  | zero =>
      constructor
      · intro h
        exact False.elim h
      · rintro ⟨j, hj, hn, _⟩
        omega
  | succ n ih =>
      rw [IterCovers, ih]
      constructor
      · intro h
        rcases h with hfirst | ⟨j, hj, hn, hx⟩
        · exact ⟨cur, le_rfl, by omega, hfirst⟩
        · exact ⟨j, by omega, by omega, hx⟩
      · rintro ⟨j, hj, hn, hx⟩
        by_cases heq : j = cur
        · subst j
          exact Or.inl hx
        · exact Or.inr ⟨j, by omega, by omega, hx⟩

theorem sourceCovers_iff (p : Parameters) (pid numCtas x : Nat) :
    SourceCovers p pid numCtas x ↔
      if p.oneTile then
        ∃ i : Fin p.block, pid * p.block + i.val = x ∧ x < p.size
      else ∃ j, j < p.iterations ∧ ∃ i : Fin p.block,
        (pid + j * numCtas) * p.block + i.val = x ∧ x < p.size := by
  cases hb : p.oneTile with
  | «true» => simp only [SourceCovers, hb, ↓reduceIte, TileCovers]
  | «false» =>
      simp only [SourceCovers, hb, Bool.false_eq_true, if_false, iterCovers_iff,
        Nat.zero_add, Nat.zero_le, true_and, TileCovers]

#axiomsClean tileMemory_cover
#axiomsClean tileMemory_frame_cover
#axiomsClean iterateMemory_contract
#axiomsClean sourceMemory_contract
#axiomsClean iterCovers_iff
#axiomsClean sourceCovers_iff
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric

import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Geometry
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GridExecution

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Grid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000

abbrev Geometry := VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.Layout
abbrev Lane (g : Geometry) := VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.Lane g
abbrev Index (g : Geometry) := {x : Nat × Lane g // x.1 < Geometry.count g ∧ Geometry.Active g x.1 x.2}

def address (g : Geometry) (x : Index g) : Nat := Geometry.offset g x.val.1 x.val.2
def owns (g : Geometry) (pid : Nat) (x : Index g) : Prop := x.val.1=pid

theorem injective (g : Geometry) (valid : Geometry.Valid g) : Function.Injective (address g) := by
  intro x y he
  have hp : x.val.1=y.val.1 := by
    by_contra h
    exact Geometry.stores_disjoint g valid x.val.1 y.val.1 x.property.1 y.property.1
      h x.val.2 y.val.2 x.property.2 y.property.2 he
  have hi : x.val.2=y.val.2 := by
    apply Geometry.offset_injective g valid x.val.1 x.val.2 y.val.2 x.property.2
    · simpa only [hp] using y.property.2
    · simpa only [address,hp] using he
  apply Subtype.ext
  exact Prod.ext hp hi

/-- The local program contract is used only internally, and instantiated below
by the proved complete source kernels. It is never a public success premise. -/
structure Program {α : Type} (g : Geometry) (output : RegionName) where
  run : Nat → State α → Option (State α)
  value : State α → Nat → Lane g → Cell α
  stable : ∀ initial current, Outside output initial current → ∀ pid i,
    value current pid i=value initial pid i
  complete : ∀ pid, pid<Geometry.count g → ∀ current,
    ∃ final, run pid current=some final ∧
      (∀ i, Geometry.Active g pid i → final.mem output (Geometry.offset g pid i)=value current pid i) ∧
      (∀ r o, (r≠output ∨ ∀ i, Geometry.Active g pid i → o≠Geometry.offset g pid i) →
        final.mem r o=current.mem r o)

def run {α : Type} {g : Geometry} {output : RegionName} (p : Program (α:=α) g output)
    (programs : List Nat) (s : State α) : Option (State α) :=
  GridExecution.sequence p.run programs s

theorem whole {α : Type} (g : Geometry) (output : RegionName) (valid : Geometry.Valid g)
    (p : Program (α:=α) g output) (initial : State α) (programs : List Nat)
    (schedule : programs.Perm (List.range (Geometry.count g))) :
    ∃ final, run p programs initial=some final ∧
      (∀ pid, pid<Geometry.count g → ∀ i, Geometry.Active g pid i →
        final.mem output (Geometry.offset g pid i)=p.value initial pid i) ∧
      (∀ r o, (r≠output ∨ ∀ pid, pid<Geometry.count g → ∀ i,
        Geometry.Active g pid i → o≠Geometry.offset g pid i) → final.mem r o=initial.mem r o) := by
  have hprograms : ∀ pid∈programs, pid<Geometry.count g := by
    intro pid hp
    exact List.mem_range.mp (schedule.mem_iff.mp hp)
  have single : ∀ pid, pid<Geometry.count g → ∀ current, Outside output initial current →
      ∃ final, p.run pid current=some final ∧
        GridExecution.Post output (address g) (fun x : Index g => p.value initial x.val.1 x.val.2)
          (owns g pid) current final := by
    intro pid hp current hout
    obtain ⟨final,he,hr,hf⟩ := p.complete pid hp current
    refine ⟨final,he,?_,?_⟩
    · intro x hx
      have hx' : x.val.1=pid := hx
      change final.mem output (Geometry.offset g x.val.1 x.val.2)=_
      rw [hx',hr x.val.2 (by simpa only [hx'] using x.property.2),p.stable initial current hout]
      simp only [hx']
    · intro r o h
      apply hf r o
      rcases h with h|h
      · exact Or.inl h
      · right
        intro i hi
        exact h ⟨(pid,i),hp,hi⟩ rfl
  obtain ⟨final,he,hr,hf⟩ := GridExecution.compose output (address g) (injective g valid)
    (fun x : Index g => p.value initial x.val.1 x.val.2) (owns g) p.run initial
    (Geometry.count g) single programs hprograms initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro pid hp i hi
    exact hr ⟨(pid,i),hp,hi⟩ ⟨pid,schedule.mem_iff.mpr (List.mem_range.mpr hp),rfl⟩
  · intro r o h
    apply hf r o
    exact h.imp id (fun h x _ => h x.val.1 x.property.1 x.val.2 x.property.2)

#axiomsClean injective
#axiomsClean whole
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Grid

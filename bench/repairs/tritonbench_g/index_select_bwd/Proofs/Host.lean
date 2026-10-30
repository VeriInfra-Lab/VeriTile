import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Native

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Host
open VeriTile.Triton FP.Structural Logical
noncomputable section

def backend {A : Type} (rest : Algebra A) (add : A → A → A) : Algebra A :=
  {rest with binary := fun _ _ _ => add}

def flat {R C : Nat} (q : Coord R C) : Nat := q.1.val*C+q.2.val

def coordinate (R C i : Nat) (hi : i<R*C) : Coord R C :=
  have positive : 0<C := by nlinarith
  (⟨i/C,(Nat.div_lt_iff_lt_mul positive).2 (by simpa only [Nat.mul_comm] using hi)⟩,
    ⟨i%C,Nat.mod_lt _ positive⟩)

theorem flat_bound {R C : Nat} (q : Coord R C) : flat q<R*C :=
  Native.flat_bound R C q.1.val q.2.val q.1.isLt q.2.isLt

theorem coordinate_flat {R C : Nat} (q : Coord R C) : coordinate R C (flat q) (flat_bound q)=q := by
  apply Prod.ext <;> apply Fin.ext
  · change (q.1.val*C+q.2.val)/C=q.1.val
    rw [Nat.mul_comm q.1.val C,Nat.mul_add_div (by have bound:=q.2.isLt; omega : 0<C),
      Nat.div_eq_of_lt q.2.isLt,Nat.add_zero]
  · simp [coordinate,flat,Nat.add_mod,Nat.mod_eq_of_lt q.2.isLt]

def extend {A : Type} (R C : Nat) (values : Coord R C → A) (original : Nat → A) (i : Nat) : A :=
  if hi : i<R*C then values (coordinate R C i hi) else original i

theorem extend_read {A : Type} {R C : Nat} (values : Coord R C → A) (original : Nat → A)
    (q : Coord R C) : extend R C values original (flat q)=values q := by
  rw [extend,dif_pos (flat_bound q),coordinate_flat]

/-- Independent derivative over the signed metadata and arbitrary Real
payloads. It contains no source execution, finite encoding, or schedule. -/
def target (r : Request) (g h : Nat → ℝ) (q : Coord r.R r.C) : ℝ :=
  g (flat q) + ((List.finRange r.N).map fun (i : Fin r.N) =>
    if r.indices i=(q.1.val : Int) then h (flat (i,q.2)) else 0).sum

/-- Independent ordered scalar reference: increasing input row, with the
caller's original destination value as the initial accumulator. -/
def orderedTarget {A : Type} (add : A → A → A) (r : Request) (g h : Nat → A)
    (q : Coord r.R r.C) : A :=
  (List.finRange r.N).foldl (fun acc (i : Fin r.N) =>
    if r.indices i=(q.1.val : Int) then add acc (h (flat (i,q.2))) else acc) (g (flat q))

structure Result (A : Type) where
  destination : Nat → A
  incoming : Nat → A
  returned : Option Unit
  object : Nat
  device : VeriTile.Torch.Device

/-- Every chunk is the complete typed source; the host submits input rows in
ascending order, then the contiguous column slices of that row. -/
def run {A : Type} [Inhabited A] (rest : Algebra A) (r : Request) (guard : InputContract r)
    (g h : Nat → A) (object : Nat) (device : VeriTile.Torch.Device) : Result A :=
  ⟨extend r.R r.C (Program.run rest (Native.selected r guard)
    (fun q => h (flat q)) (fun q => g (flat q))) g,h,none,object,device⟩

theorem source_readback {A : Type} [Inhabited A] (rest : Algebra A) (r : Request)
    (guard : InputContract r) (g h : Nat → A) (object : Nat) (device : VeriTile.Torch.Device)
    (q : Coord r.R r.C) :
    (run rest r guard g h object device).destination (flat q)=
      rows (Source.scalar rest) (Native.selected r guard) (fun q => h (flat q))
        (List.finRange r.N) (fun q => g (flat q)) q := by
  simp only [run]
  rw [extend_read,Program.run_correct]

theorem mathematical_real (rest : Algebra ℝ) (r : Request) (guard : InputContract r)
    (g h : Nat → ℝ) (object : Nat) (device : VeriTile.Torch.Device) (q : Coord r.R r.C) :
    (run (backend rest (·+·)) r guard g h object device).destination (flat q)=target r g h q := by
  rw [source_readback]
  change rows (·+·) _ _ _ _ q=_
  rw [complete_real]
  unfold Logical.target target
  congr 2
  apply List.map_congr_left
  intro i _
  have same := (Native.selected_value r guard i q.2).2
  have iff : Native.selected r guard i=q.1 ↔ r.indices i=(q.1.val : Int) := by
    constructor
    · intro eq; rw [←same,eq]
    · intro eq; apply Fin.ext; omega
  simp only [iff]

theorem ordered_readback {A : Type} [Inhabited A] (rest : Algebra A) (r : Request)
    (guard : InputContract r) (g h : Nat → A) (object : Nat) (device : VeriTile.Torch.Device)
    (q : Coord r.R r.C) :
    (run rest r guard g h object device).destination (flat q)=orderedTarget (Source.scalar rest) r g h q := by
  rw [source_readback,rows_cell]
  unfold orderedTarget
  congr 1
  funext acc i
  have same := (Native.selected_value r guard i q.2).2
  have iff : Native.selected r guard i=q.1 ↔ r.indices i=(q.1.val : Int) := by
    constructor
    · intro eq; rw [←same,eq]
    · intro eq; apply Fin.ext; omega
  simp only [iff]

theorem exterior_frame {A : Type} [Inhabited A] (rest : Algebra A) (r : Request)
    (guard : InputContract r) (g h : Nat → A) (object : Nat) (device : VeriTile.Torch.Device)
    (i : Nat) (outside : r.R*r.C ≤ i) : (run rest r guard g h object device).destination i=g i := by
  simp only [run,extend,dif_neg (by omega : ¬i<r.R*r.C)]

theorem unselected_frame {A : Type} [Inhabited A] (rest : Algebra A) (r : Request)
    (guard : InputContract r) (g h : Nat → A) (object : Nat) (device : VeriTile.Torch.Device)
    (q : Coord r.R r.C) (miss : ∀ i : Fin r.N,r.indices i≠(q.1.val : Int)) :
    (run rest r guard g h object device).destination (flat q)=g (flat q) := by
  rw [source_readback]
  apply Logical.unselected
  intro i _ eq
  apply miss i
  rw [←(Native.selected_value r guard i q.2).2,eq]

theorem api {A : Type} [Inhabited A] (rest : Algebra A) (r : Request)
    (guard : InputContract r) (g h : Nat → A) (object : Nat) (device : VeriTile.Torch.Device) :
    let result := run rest r guard g h object device
    result.incoming=h ∧ result.returned=none ∧ result.object=object ∧ result.device=device :=
  ⟨rfl,rfl,rfl,rfl⟩

#axiomsClean source_readback
#axiomsClean mathematical_real
#axiomsClean ordered_readback
#axiomsClean exterior_frame
#axiomsClean unselected_frame
#specNonCircular target avoiding [run,Program.run,Source.execution]
#specNonCircular orderedTarget avoiding [run,Program.run,Source.execution]
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Host

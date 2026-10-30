import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem source_bounds (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (xs : Inputs α) (bounds : RegionName → Nat)
    (hb : (programIO native a g p).InBounds bounds xs) (hc : 0<Grid.count p)
    (initial : State α) : SourceSafety.Bounds p initial bounds := by
  have hc0 : Grid.count p≠0 := by omega
  have hm : 0<p.rows := by
    by_contra hn
    have hz : p.rows=0 := by omega
    rw [Grid.empty p hz] at hc
    omega
  have hn : 0<p.columns := by
    by_contra hn
    have hz : p.columns=0 := by omega
    simp [Grid.count,columnBlocks,hz] at hc
  have ha (o : Nat) (ho : o<p.rows*p.inner) : o<bounds "A" := by
    exact hb.2.1 (0 : Fin 7) o (by simp [programIO,reads,hc0,ho])
  have hw1 (o : Nat) (ho : o<p.columns*p.inner) : o<bounds "W1" := by
    cases hbyte : p.byte
    · exact hb.2.1 (1 : Fin 7) o (by simp [programIO,reads,hc0,hbyte,ho])
    · exact hb.2.1 (4 : Fin 7) o (by simp [programIO,reads,hc0,hbyte,ho])
  have hw3 (o : Nat) (ho : o<p.columns*p.inner) : o<bounds "W3" := by
    cases hbyte : p.byte
    · exact hb.2.1 (2 : Fin 7) o (by simp [programIO,reads,hc0,hbyte,ho])
    · exact hb.2.1 (5 : Fin 7) o (by simp [programIO,reads,hc0,hbyte,ho])
  have hr (o : Nat) (ho : o<p.inner) : o<bounds "RMS" := by
    cases hbyte : p.byte
    · exact hb.2.1 (3 : Fin 7) o (by simp [programIO,reads,hc0,hbyte,ho])
    · exact hb.2.1 (6 : Fin 7) o (by simp [programIO,reads,hc0,hbyte,ho])
  have hrow (i : Fin 16) : inputRow p initial i < p.rows := Nat.mod_lt _ hm
  have hcol (i : Fin 16) : inputColumn p initial i < p.columns := Nat.mod_lt _ hn
  have hw (k : Nat) (i : TileIndex [64,16]) (hk : activeK p k i.1.val) :
      wOffset p initial k i < p.columns*p.inner := by
    calc
      wOffset p initial k i < p.inner+inputColumn p initial i.2.1*p.inner := Nat.add_lt_add_right hk _
      _ = (inputColumn p initial i.2.1+1)*p.inner := by ring
      _ ≤ p.columns*p.inner := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt (hcol i.2.1))
  constructor
  · intro k i hk
    apply ha
    calc
      aOffset p initial k i < inputRow p initial i.1*p.inner+p.inner := Nat.add_lt_add_left hk _
      _ = (inputRow p initial i.1+1)*p.inner := by ring
      _ ≤ p.rows*p.inner := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt (hrow i.1))
  · intro k i hk; exact hw1 _ (hw k i hk)
  · intro k i hk; exact hw3 _ (hw k i hk)
  · intro k i hk; exact hr _ hk
  · intro i hi
    exact hb.2.2.1 (0 : Fin 1)
      (⟨outputRow p initial i.1,hi.1⟩,⟨outputColumn p initial i.2.1,hi.2⟩,()) trivial

/-- Concrete capacities witness that the public finite-footprint bounds are
satisfiable, including both zero-dimensional launches and K tails. -/
def capacities (p : Parameters) (r : RegionName) : Nat :=
  if r="A" then p.rows*p.inner else
  if r="W1" ∨ r="W3" then p.columns*p.inner else
  if r="RMS" then p.inner else
  if r="OUT" then p.rows*p.columns else 0

theorem capacity_bounds (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (xs : Inputs α) : (programIO native a g p).InBounds (capacities p) xs := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r hr
    simp only [programIO,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
    simp [capacities,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2]
  · intro i o ho
    have hil : i.val<7 := i.isLt
    fin_cases i <;> simp [programIO,reads,capacities,inputPorts] at ho ⊢
    all_goals tauto
  · intro o j _
    have ho : o=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance o 0
    subst o
    change p.columns*j.1.val+j.2.1.val < p.rows*p.columns
    calc
      _ < p.columns*j.1.val+p.columns := Nat.add_lt_add_left j.2.1.isLt _
      _ = (j.1.val+1)*p.columns := by ring
      _ ≤ p.rows*p.columns := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt j.1.isLt)
  · intro i; exact Fin.elim0 i

end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface

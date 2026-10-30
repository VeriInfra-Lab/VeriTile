import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHComplete
import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelFold

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000

def trajectory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed : Bool)
    (s : State α) (n : Nat) : Accum c α :=
  KernelFold.sequenceValue (advance M D c s) ⟨initValue M c seed s,s⟩ n

theorem accumulated_trajectory (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed : Bool) (s : State α) :
    accumulated M D c seed s=trajectory M D c seed s c.chunks :=
  KernelFold.from_zero _ _ _

theorem index_ext (c : Shape) (s : State α) (i j : Index c)
    (he : feature c s i.1.val=feature c s j.1.val)
    (hv : column c s i.2.1.val=column c s j.2.1.val) : i=j := by
  have h1 : i.1=j.1 := Fin.ext (by simpa [feature] using he)
  have h2 : i.2.1=j.2.1 := Fin.ext (by simpa [column] using hv)
  rcases i with ⟨ik,iv,⟨⟩⟩
  rcases j with ⟨jk,jv,⟨⟩⟩
  simp_all

theorem state_injective (c : Shape) (s : State α) (i j : Index c)
    (hi : active c s i) (hj : active c s j)
    (he : stateAddress c s i=stateAddress c s j) : i=j := by
  have hh : feature c s i.1.val*c.V+column c s i.2.1.val=
      feature c s j.1.val*c.V+column c s j.2.1.val := by
    simpa only [stateAddress,Nat.add_assoc,Nat.add_left_cancel_iff] using he
  obtain ⟨hk,hv⟩ := digit_injective c.V _ _ _ _ hi.2 hj.2 hh
  exact index_ext c s i j hk hv

theorem history_injective (c : Shape) (s : State α) (n m : Nat) (i j : Index c)
    (hi : active c s i) (hj : active c s j)
    (he : historyAddress c s n i=historyAddress c s m j) : n=m ∧ i=j := by
  have hh : (n*c.K+feature c s i.1.val)*c.V+column c s i.2.1.val=
      (m*c.K+feature c s j.1.val)*c.V+column c s j.2.1.val := by
    simpa only [historyAddress,Nat.add_mul,Nat.add_assoc,Nat.add_left_cancel_iff] using he
  obtain ⟨hk,hv⟩ := digit_injective c.V _ _ _ _ hi.2 hj.2 hh
  obtain ⟨hn,hk⟩ := digit_injective c.K _ _ _ _ hi.1 hj.1 hk
  exact ⟨hn,index_ext c s i j hk hv⟩

theorem stored_read (M : Algebra α) (c : Shape) (initial current : State α)
    (n : Nat) (carry : Index c → α) (i : Index c) (hi : active c initial i) :
    (stored M c initial current n carry).mem "h" (historyAddress c initial n i)=
      .mk .real (M.cast (some .fp32) .real .real (carry i)) := by
  apply masked_readback_active _ _ _ _ _ i hi
  intro a b ha hb he
  exact (history_injective c initial n n a b ha hb he).2

theorem stored_frame (M : Algebra α) (c : Shape) (initial current : State α)
    (n : Nat) (carry : Index c → α) (r : RegionName) (off : Nat)
    (h : r≠"h" ∨ ∀ i, active c initial i → off≠historyAddress c initial n i) :
    (stored M c initial current n carry).mem r off=current.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i hi)

theorem trajectory_history (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed : Bool) (s : State α) (n k : Nat) (i : Index c)
    (hk : k<n) (hi : active c s i) :
    (trajectory M D c seed s n).2.mem "h" (historyAddress c s k i)=
      .mk .real (M.cast (some .fp32) .real .real ((trajectory M D c seed s k).1 i)) := by
  induction n with
  | zero => omega
  | succ n ih =>
    change (stored M c s (trajectory M D c seed s n).2 n
      (trajectory M D c seed s n).1).mem "h" (historyAddress c s k i)=_
    by_cases he : k=n
    · subst k
      exact stored_read M c s _ n _ i hi
    · rw [stored_frame M c s _ n _ "h" _ (Or.inr (by
        intro j hj eq
        exact he (history_injective c s k n i j hi hj eq).1))]
      exact ih (by omega)

theorem trajectory_frame (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed : Bool) (s : State α) (n : Nat) (r : RegionName) (off : Nat)
    (h : r≠"h" ∨ ∀ k, k<n → ∀ i, active c s i → off≠historyAddress c s k i) :
    (trajectory M D c seed s n).2.mem r off=s.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (stored M c s (trajectory M D c seed s n).2 n
      (trajectory M D c seed s n).1).mem r off=_
    rw [stored_frame M c s _ n _ r off (h.imp_right (fun hh => hh n (by omega)))]
    exact ih (h.imp_right (fun hh k hk => hh k (by omega)))

theorem history_readback (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed final : Bool) (s : State α) (k : Nat) (i : Index c)
    (hk : k<c.chunks) (hi : active c s i) :
    (written M D c seed final s).mem "h" (historyAddress c s k i)=
      .mk .real (M.cast (some .fp32) .real .real ((trajectory M D c seed s k).1 i)) := by
  have hf : ∀ current carry, (finalStored M c s current carry).mem "h"=
      current.mem "h" := by
    intro current carry
    funext off
    apply State.masked_scatter_frame
    exact Or.inl (by decide)
  cases final <;> simp only [written,Bool.false_eq_true,↓reduceIte]
  · rw [accumulated_trajectory]
    exact trajectory_history M D c seed s c.chunks k i hk hi
  · rw [hf,accumulated_trajectory]
    exact trajectory_history M D c seed s c.chunks k i hk hi

theorem final_readback (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed : Bool) (s : State α) (i : Index c) (hi : active c s i) :
    (written M D c seed Bool.true s).mem "ht" (stateAddress c s i)=
      .mk .real (M.cast (some .fp32) .real .real ((trajectory M D c seed s c.chunks).1 i)) := by
  simp only [written,↓reduceIte,accumulated_trajectory]
  apply masked_readback_active _ _ _ _ _ i hi
  exact state_injective c s

theorem written_frame (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed final : Bool) (s : State α) (r : RegionName) (off : Nat)
    (hh : r≠"h" ∨ ∀ k, k<c.chunks → ∀ i, active c s i → off≠historyAddress c s k i)
    (hf : final=Bool.false ∨ r≠"ht" ∨ ∀ i, active c s i → off≠stateAddress c s i) :
    (written M D c seed final s).mem r off=s.mem r off := by
  have hmem : (accumulated M D c seed s).2.mem r off=s.mem r off := by
    rw [accumulated_trajectory]
    exact trajectory_frame M D c seed s c.chunks r off hh
  cases final
  · exact hmem
  ·
    change (finalStored M c s _ _).mem r off=_
    rw [show (finalStored M c s (accumulated M D c seed s).2
      (accumulated M D c seed s).1).mem r off=
      (accumulated M D c seed s).2.mem r off from by
        apply State.masked_scatter_frame
        rcases hf with hfalse|hr|ho
        · cases hfalse
        · exact Or.inl hr
        · exact Or.inr (fun i _ hi => ho i hi)]
    exact hmem

#axiomsClean history_readback
#axiomsClean final_readback
#axiomsClean written_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource

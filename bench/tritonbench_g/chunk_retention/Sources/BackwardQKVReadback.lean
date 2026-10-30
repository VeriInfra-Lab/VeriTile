import bench.tritonbench_g.chunk_retention.Sources.BackwardQKVComplete

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
open VeriTile Triton FP.Structural KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 5000000

theorem key_injective (c : Shape) (s : State α) (i j : KeyIndex c)
    (hi : keyActive c s i) (hj : keyActive c s j)
    (he : keyAddress c s i=keyAddress c s j) : i=j := by
  have hh : row s i.1.val*c.K+feature c s i.2.1.val=
      row s j.1.val*c.K+feature c s j.2.1.val := by
    simpa only [keyAddress,Nat.add_assoc,Nat.add_left_cancel_iff] using he
  obtain ⟨hr,hk⟩ := digit_injective c.K _ _ _ _ hi.2 hj.2 hh
  have h1 : i.1=j.1 := Fin.ext (by simpa [row] using hr)
  have h2 : i.2.1=j.2.1 := Fin.ext (by simpa [feature] using hk)
  rcases i with ⟨ik,iv,⟨⟩⟩
  rcases j with ⟨jk,jv,⟨⟩⟩
  simp_all

theorem partial_injective (c : Shape) (s : State α) (f g : Nat) (i j : ValueIndex c)
    (hi : valueActive c s f i) (hj : valueActive c s g j)
    (he : partialAddress c s f i=partialAddress c s g j) : f=g ∧ i=j := by
  have hh : row s i.1.val*c.V+column c f i.2.1.val=
      row s j.1.val*c.V+column c g j.2.1.val := by
    simpa only [partialAddress,Nat.add_assoc,Nat.add_left_cancel_iff] using he
  obtain ⟨hr,hv⟩ := digit_injective c.V _ _ _ _ hi.2 hj.2 hh
  obtain ⟨hf,hv⟩ := digit_injective c.BV f i.2.1.val g j.2.1.val i.2.1.isLt j.2.1.isLt hv
  have h1 : i.1=j.1 := Fin.ext (by simpa [row] using hr)
  have h2 : i.2.1=j.2.1 := Fin.ext hv
  refine ⟨hf,?_⟩
  rcases i with ⟨ik,iv,⟨⟩⟩
  rcases j with ⟨jk,jv,⟨⟩⟩
  simp_all

theorem partial_read (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (initial current : State α) (scale : ℝ) (f : Nat) (i : ValueIndex c)
    (hi : valueActive c initial f i) :
    (storedPartial M D c initial current scale f).mem "dv" (partialAddress c initial f i)=
      .mk .real (partialValue M D c initial scale f i) := by
  apply masked_readback_active _ _ _ _ _ i hi
  intro a b ha hb he
  exact (partial_injective c initial f f a b ha hb he).2

theorem partial_frame (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (initial current : State α) (scale : ℝ) (f : Nat) (r : RegionName) (off : Nat)
    (h : r≠"dv" ∨ ∀ i, valueActive c initial f i → off≠partialAddress c initial f i) :
    (storedPartial M D c initial current scale f).mem r off=current.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i hi)

theorem key_read (c : Shape) (initial current : State α) (r : RegionName)
    (value : KeyIndex c → α) (i : KeyIndex c) (hi : keyActive c initial i) :
    (storedKey c initial current r value).mem r (keyAddress c initial i)=.mk .real (value i) :=
  masked_readback_active _ _ _ _ _ i hi (key_injective c initial)

theorem key_frame (c : Shape) (initial current : State α) (r : RegionName)
    (value : KeyIndex c → α) (r' : RegionName) (off : Nat)
    (h : r'≠r ∨ ∀ i, keyActive c initial i → off≠keyAddress c initial i) :
    (storedKey c initial current r value).mem r' off=current.mem r' off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i hi)

theorem trajectory_partial (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (n f : Nat) (i : ValueIndex c)
    (hf : f<n) (hi : valueActive c s f i) :
    (trajectory M D c s scale n).2.mem "dv" (partialAddress c s f i)=
      .mk .real (partialValue M D c s scale f i) := by
  induction n with
  | zero => omega
  | succ n ih =>
    change (storedPartial M D c s (trajectory M D c s scale n).2 scale n).mem "dv" (partialAddress c s f i)=_
    by_cases he : f=n
    · subst f
      exact partial_read M D c s _ scale n i hi
    · rw [partial_frame M D c s _ scale n "dv" _ (Or.inr (by
        intro j hj eq
        exact he (partial_injective c s f n i j hi hj eq).1))]
      exact ih (by omega)

theorem trajectory_frame (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (n : Nat) (r : RegionName) (off : Nat)
    (h : r≠"dv" ∨ ∀ f, f<n → ∀ i, valueActive c s f i → off≠partialAddress c s f i) :
    (trajectory M D c s scale n).2.mem r off=s.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (storedPartial M D c s (trajectory M D c s scale n).2 scale n).mem r off=_
    rw [partial_frame M D c s _ scale n r off (h.imp_right (fun hh => hh n (by omega)))]
    exact ih (h.imp_right (fun hh f hf => hh f (by omega)))

theorem partial_readback (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (f : Nat) (i : ValueIndex c)
    (hf : f<c.valueBlocks) (hi : valueActive c s f i) :
    (written M D c s scale).mem "dv" (partialAddress c s f i)=
      .mk .real (partialValue M D c s scale f i) := by
  unfold written finishStored
  rw [key_frame c s _ "dk" _ "dv" _ (Or.inl (by decide)),
    key_frame c s _ "dq" _ "dv" _ (Or.inl (by decide))]
  exact trajectory_partial M D c s scale c.valueBlocks f i hf hi

theorem query_readback (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (i : KeyIndex c) (hi : keyActive c s i) :
    (written M D c s scale).mem "dq" (keyAddress c s i)=
      .mk .real (queryValue M D c s scale (trajectory M D c s scale c.valueBlocks).1 i) := by
  unfold written finishStored
  rw [key_frame c s _ "dk" _ "dq" _ (Or.inl (by decide))]
  exact key_read c s _ "dq" _ i hi

theorem key_readback (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (i : KeyIndex c) (hi : keyActive c s i) :
    (written M D c s scale).mem "dk" (keyAddress c s i)=
      .mk .real (keyValue M D c s scale (trajectory M D c s scale c.valueBlocks).1 i) :=
  key_read c s _ "dk" _ i hi

theorem frame (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (r : RegionName) (off : Nat)
    (hq : r≠"dq" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i)
    (hk : r≠"dk" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i)
    (hv : r≠"dv" ∨ ∀ f, f<c.valueBlocks → ∀ i, valueActive c s f i → off≠partialAddress c s f i) :
    (written M D c s scale).mem r off=s.mem r off := by
  unfold written finishStored
  rw [key_frame c s _ "dk" _ r off hk,key_frame c s _ "dq" _ r off hq]
  exact trajectory_frame M D c s scale c.valueBlocks r off hv

#axiomsClean partial_readback
#axiomsClean query_readback
#axiomsClean key_readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource

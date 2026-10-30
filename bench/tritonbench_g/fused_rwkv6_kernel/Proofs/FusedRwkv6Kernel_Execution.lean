import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Index
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 8000000
set_option maxRecDepth 10000
variable {α : Type}

@[simp] theorem storePartial_input (a : Ops α) (c : Shape) (p : Program) (n : Nat)
    (v : Nat → α) (m : Memory α) (r : Region) (hr : InputRegion r) (b : Nat) :
    storePartial a c p n v m r b=m r b := by
  have : r≠.o := by rcases hr with rfl|rfl|rfl|rfl|rfl|rfl <;> decide
  simp [storePartial,this]

@[simp] theorem storeFinal_input (a : Ops α) (c : Shape) (p : Program)
    (h : Carry α) (m : Memory α) (r : Region) (hr : InputRegion r) (b : Nat) :
    storeFinal a c p h m r b=m r b := by
  have : r≠.ht := by rcases hr with rfl|rfl|rfl|rfl|rfl|rfl <;> decide
  simp [storeFinal,this]

theorem loop_inputs (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (scale : α)
    (m : Memory α) (n : Nat) : InputsEq (runLoop a c plan p scale m n).memory m := by
  intro r hr b
  induction n with
  | zero => rfl
  | succ n ih => simpa [runLoop,step,storePartial_input a c p n _ _ r hr] using ih

theorem program_inputs (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (scale : α)
    (m : Memory α) : InputsEq (runProgram a c plan p scale m) m := by
  intro r hr b
  simp only [runProgram,storeFinal_input a c p _ _ r hr]
  exact loop_inputs a c plan p scale m c.T r hr b

theorem initial_ref (a : Ops α) (c : Shape) (p : Program) (m : Memory α) :
    initialCarry a c p m=fun l j => scalarState a c (inputs c m) p.z
      (keyIndex c p j) (valueIndex c p l) 0 := by
  funext l j
  simp [initialCarry,scalarState,scalarSeed,load,inputs]

theorem next_ref (a : Ops α) (c : Shape) (p : Program) (m : Memory α) (n : Nat) :
    nextCarry a c p m (fun l j => scalarState a c (inputs c m) p.z
      (keyIndex c p j) (valueIndex c p l) n) n =
    fun l j => scalarState a c (inputs c m) p.z (keyIndex c p j) (valueIndex c p l) (n+1) := by
  funext l j
  simp [nextCarry,scalarState,decay,outer,load,inputs,scalarOuter]

theorem loop_carry (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (scale : α)
    (m : Memory α) (n : Nat) :
    (runLoop a c plan p scale m n).carry=fun l j =>
      scalarState a c (inputs c m) p.z (keyIndex c p j) (valueIndex c p l) n := by
  induction n with
  | zero => exact initial_ref a c p m
  | succ n ih =>
    simp only [runLoop,step]
    rw [ih]
    have hx := inputs_eq c (loop_inputs a c plan p scale m n)
    rw [←hx,next_ref,hx]

theorem partial_ref (a : Ops α) (c : Shape) (plan : Plans c) (p : Program)
    (m : Memory α) (scale : α) (n l : Nat) :
    partialValue a c plan p m scale (fun l j => scalarState a c (inputs c m) p.z
      (keyIndex c p j) (valueIndex c p l) n) n l =
    scalarPartial a c plan (inputs c m) scale p.z p.ik n (valueIndex c p l) := by
  simp [partialValue,scalarPartial,scalarTerm,outer,bias,query,load,inputs,scalarOuter,keyIndex]

theorem loop_partial (a : Ops α) (c : Shape) (plan : Plans c) (p : Program)
    (m : Memory α) (scale : α) (n l : Nat) :
    partialValue a c plan p (runLoop a c plan p scale m n).memory scale
      (runLoop a c plan p scale m n).carry n l =
    scalarPartial a c plan (inputs c m) scale p.z p.ik n (valueIndex c p l) := by
  rw [loop_carry]
  have hx := inputs_eq c (loop_inputs a c plan p scale m n)
  rw [←hx,partial_ref,hx]

theorem storePartial_target (a : Ops α) (c : Shape) (plan : Plans c) (p : Program)
    (hp : Valid c p) (x : Input α) (scale : α) (m : Memory α) (n : Nat) (hn : n<c.T)
    (v : Nat → α) (hv : ∀ l, l<c.BV → valueIndex c p l<c.V →
      v l=scalarPartial a c plan x scale p.z p.ik n (valueIndex c p l)) (r : Region) (b : Nat) :
    storePartial a c p n v m r b =
      if PartialWrite c p n r b then target a c plan x scale r b else m r b := by
  simp only [PartialWrite]
  unfold storePartial
  split
  · rename_i hw
    rcases hw with ⟨rfl,l,hl,hf,rfl⟩
    have dec := partial_decode c p.z p.ik (visit c n) (valueIndex c p l) hp.1 (visit_lt c hn) hf
    have hw : PartialWrite c p n .o (partialAddress c p.z p.ik (visit c n) (valueIndex c p l)) :=
      ⟨rfl,l,hl,hf,rfl⟩
    have hw := (show (Region.o=Region.o ∧ ∃ l1<c.BV, valueIndex c p l1<c.V ∧
      partialAddress c p.z p.ik (visit c n) (valueIndex c p l)=
      partialAddress c p.z p.ik (visit c n) (valueIndex c p l1)) from hw)
    simp only [if_pos hw,partial_lane,hv l hl hf,
      target,dec.1,dec.2.1,dec.2.2.1,dec.2.2.2,visit_visit c hn]
  · rename_i hw
    simp only [if_neg hw]

theorem storeFinal_target (a : Ops α) (c : Shape) (plan : Plans c) (p : Program)
    (x : Input α) (scale : α) (m : Memory α) (h : Carry α)
    (hh : ∀ l j, h l j=scalarState a c x p.z (keyIndex c p j) (valueIndex c p l) c.T)
    (r : Region) (b : Nat) : storeFinal a c p h m r b=
      if FinalWrite c p r b then target a c plan x scale r b else m r b := by
  simp only [FinalWrite]
  unfold storeFinal
  split
  · rename_i hw
    rcases hw with ⟨hs,rfl,l,hl,j,hj,hf,he,rfl⟩
    have dec := state_decode c p.z (keyIndex c p j) (valueIndex c p l) he hf
    have hw : FinalWrite c p .ht (stateAddress c p.z (keyIndex c p j) (valueIndex c p l)) :=
      ⟨hs,rfl,l,hl,j,hj,hf,he,rfl⟩
    unfold FinalWrite at hw
    simp only [if_pos hw,target,dec.1,dec.2.1,dec.2.2]
    have hlane : valueIndex c p l-p.iv*c.BV=l := by unfold valueIndex;omega
    have jlane : keyIndex c p j-p.ik*c.BK=j := by unfold keyIndex;omega
    rw [hlane,jlane,hh]
  · rename_i hw
    simp only [if_neg hw]

theorem loop_memory (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (hp : Valid c p)
    (m : Memory α) (scale : α) (n : Nat) (hn : n≤c.T) (r : Region) (b : Nat) :
    (runLoop a c plan p scale m n).memory r b =
    if ∃ t<n, PartialWrite c p t r b then target a c plan (inputs c m) scale r b else m r b := by
  induction n with
  | zero => simp [runLoop]
  | succ n ih =>
    simp only [runLoop,step]
    rw [storePartial_target a c plan p hp (inputs c m) scale _ n (by omega) _
      (fun l _ _ => loop_partial a c plan p m scale n l) r b]
    rw [ih (by omega)]
    have heq : (∃ t<n+1, PartialWrite c p t r b) ↔
        PartialWrite c p n r b ∨ ∃ t<n, PartialWrite c p t r b := by
      constructor
      · rintro ⟨t,ht,h⟩
        by_cases he : t=n
        · exact Or.inl (he ▸ h)
        · exact Or.inr ⟨t,by omega,h⟩
      · rintro (h|⟨t,ht,h⟩)
        · exact ⟨n,by omega,h⟩
        · exact ⟨t,by omega,h⟩
    simp only [heq]
    by_cases h : PartialWrite c p n r b <;> simp [h]

theorem program_memory (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (hp : Valid c p)
    (m : Memory α) (scale : α) (r : Region) (b : Nat) :
    runProgram a c plan p scale m r b =
    if Writes c p r b then target a c plan (inputs c m) scale r b else m r b := by
  unfold runProgram
  rw [storeFinal_target a c plan p (inputs c m) scale _ _
    (fun l j => congrFun (congrFun (loop_carry a c plan p scale m c.T) l) j)]
  rw [loop_memory a c plan p hp m scale c.T le_rfl]
  unfold Writes
  by_cases h : FinalWrite c p r b <;> simp [h]

theorem grid_inputs (a : Ops α) (c : Shape) (plan : Plans c) (scale : α)
    (ps : List Program) (m : Memory α) : InputsEq (runGrid a c plan scale ps m) m := by
  induction ps generalizing m with
  | nil => intro r hr b;rfl
  | cons p ps ih =>
    intro r hr b
    simp only [runGrid,List.foldl_cons]
    exact (ih _ r hr b).trans (program_inputs a c plan p scale m r hr b)

theorem grid_memory (a : Ops α) (c : Shape) (plan : Plans c) (scale : α)
    (ps : List Program) (valid : ∀ p∈ps, Valid c p) (m : Memory α) (r : Region) (b : Nat) :
    runGrid a c plan scale ps m r b =
    if ∃ p∈ps, Writes c p r b then target a c plan (inputs c m) scale r b else m r b := by
  induction ps generalizing m with
  | nil => simp [runGrid]
  | cons p ps ih =>
    simp only [runGrid,List.foldl_cons]
    change runGrid a c plan scale ps (runProgram a c plan p scale m) r b = _
    rw [ih (fun p h => valid p (by simp [h]))]
    rw [inputs_eq c (program_inputs a c plan p scale m)]
    rw [program_memory a c plan p (valid p (by simp)) m scale r b]
    have heq : (∃ q∈p::ps, Writes c q r b) ↔
        (∃ q∈ps, Writes c q r b) ∨ Writes c p r b := by simp [or_comm]
    simp only [heq]
    by_cases h : ∃ q∈ps, Writes c q r b <;> simp [h]

/-- Complete launched grid: no successful execution or output relation is assumed. -/
theorem complete_grid (a : Ops α) (c : Shape) (hc : Sizes c) (plan : Plans c) (scale : α)
    (m : Memory α) (r : Region) (b : Nat) :
    runGrid a c plan scale (programs c) m r b =
      if Output c r b then target a c plan (inputs c m) scale r b else m r b := by
  rw [grid_memory a c plan scale (programs c) (fun p h => (mem_programs c p).1 h)]
  have h : (∃ p∈programs c, Writes c p r b) ↔ Output c r b := by
    constructor
    · rintro ⟨p,hp,hw⟩
      exact writes_output c p ((mem_programs c p).1 hp) hw
    · exact cover c hc
  simp only [h]

end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel

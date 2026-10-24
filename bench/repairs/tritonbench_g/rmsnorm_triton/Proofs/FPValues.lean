import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.FPSource

namespace VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems.FPLoop
open VeriTile.Bench.Repairs.SoftmaxFlaggems.FPReference (scatter)
noncomputable section
set_option maxHeartbeats 4000000

/-- Logical recurrence over input values. It contains no kernel, interpreter,
registers, or memory state, and retains every source floating operation. -/
def laneStep (n B : Nat) (xs : Nat → Term) (cur : Nat)
    (v : Values Term .real [B]) : Values Term .real [B] := fun i =>
  add (v i) (square (cast32 (if cur+i.1.val<n then xs (cur+i.1.val) else lit 0)))

def logicalStd (n B : Nat) (eps : ℝ) (xs : Nat → Term) : Term :=
  let v := (walk (laneStep n B xs) n B 0 (fun _=>lit 0)).2
  rsqrt (add (divide (.sum (some .fp32) B (fun j=>v (j,PUnit.unit))) (lit n)) (lit eps))

def logical (n B : Nat) (eps : ℝ) (xs ws : Nat → Term) (k : Nat) : Term :=
  mul (mul (cast32 (xs k)) (logicalStd n B eps xs)) (ws k)

theorem logical_congr (n B : Nat) (eps : ℝ) (xs ys ws vs : Nat → Term)
    (hx : ∀ k,k<n → xs k=ys k) (hw : ∀ k,k<n → ws k=vs k) (k : Nat) (hk : k<n) :
    logical n B eps xs ws k=logical n B eps ys vs k := by
  have hstep : laneStep n B xs=laneStep n B ys := by
    funext cur v i
    simp only [laneStep]
    split <;> simp_all
  simp only [logical,logicalStd,hstep,hx k hk,hw k hk]

theorem walk_invariant {β : Type} (f : Nat → β → β) (stop stride : Nat)
    (positive : 0<stride) (Inv : Nat → β → Prop)
    (step : ∀ cur v,cur<stop → Inv cur v → Inv (cur+stride) (f cur v))
    (cur : Nat) (v : β) (hi : Inv cur v) :
    stop≤(walk f stop stride cur v).1 ∧ Inv (walk f stop stride cur v).1 (walk f stop stride cur v).2 := by
  rw [walk,if_neg (by omega)]
  by_cases hc : cur<stop
  · rw [if_pos hc]
    exact walk_invariant f stop stride positive Inv step (cur+stride) (f cur v) (step cur v hc hi)
  · rw [if_neg hc]
    exact ⟨by omega,hi⟩
termination_by stop-cur

theorem masked_preserves {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active] (l : List ι)
    (s : State α) (o : Nat) (v : Cell α) (hs : s.mem r o=v)
    (agree : ∀ i∈l,active i → off i=o → val i=v) :
    (l.foldl (fun t i=>if active i then t.write r (off i) (val i) else t) s).mem r o=v := by
  induction l generalizing s with
  | nil => exact hs
  | cons i rest ih =>
    apply ih
    · by_cases ha : active i
      · simp only [if_pos ha,State.write]
        split
        · rename_i hit
          exact agree i List.mem_cons_self ha hit.2.symm
        · exact hs
      · simpa [ha] using hs
    · intro j hj
      exact agree j (List.mem_cons_of_mem _ hj)

theorem masked_readback_active {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active] (l : List ι)
    (s : State α) (i : ι) (hi : i∈l) (ha : active i)
    (agree : ∀ j∈l,active j → off j=off i → val j=val i) :
    (l.foldl (fun t j=>if active j then t.write r (off j) (val j) else t) s).mem r (off i)=val i := by
  induction l generalizing s with
  | nil => simp at hi
  | cons j rest ih =>
    rw [List.foldl_cons]
    by_cases hr : i∈rest
    · exact ih _ hr (fun k hk=>agree k (List.mem_cons_of_mem _ hk))
    · have eq : i=j := (List.mem_cons.mp hi).resolve_right hr
      subst j
      simp only [if_pos ha]
      exact masked_preserves r off val active rest _ _ _ (State.write_same _ _ _ _)
        (fun k hk=>agree k (List.mem_cons_of_mem _ hk))

def RowUnique (p : Parameters) (b m : Nat) : Prop :=
  ∀ k j,k<p.width → j<p.width → TensorSource.rowAddress p b m k=TensorSource.rowAddress p b m j → k=j

theorem emit_frame (p : Parameters) (b m B : Nat) (std : Term) (cur : Nat) (s : State Term)
    (r : RegionName) (o : Nat)
    (outside : r≠"out" ∨ ∀ i : TileIndex [B],cur+i.1.val<p.width → o≠TensorSource.rowAddress p b m (cur+i.1.val)) :
    (emit p b m B std cur s).mem r o=s.mem r o := by
  unfold emit scatter
  apply State.masked_scatter_frame
  rcases outside with hr | ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi=>ho i (by simpa using hi))

theorem emit_value (p : Parameters) (b m B : Nat) (std : Term) (cur : Nat) (s : State Term)
    (unique : RowUnique p b m) (k : Nat) (hk : k<p.width) (lo : cur≤k) (hi : k<cur+B) :
    (emit p b m B std cur s).mem "out" (TensorSource.rowAddress p b m k)=
      .mk .real (storedValue s p b m k std) := by
  let i : TileIndex [B] := (⟨k-cur,by omega⟩,PUnit.unit)
  have eq : cur+i.1.val=k := by dsimp [i]; omega
  unfold emit scatter
  rw [←eq]
  apply masked_readback_active
  · exact TileShape.mem_allIndices _ _
  · simp [eq,hk]
  · intro j _ hj same
    have samek := unique _ _ (by simpa using hj) (by simpa [eq] using hk) same
    simp only [samek]

def WriteInvariant (base : State Term) (p : Parameters) (b m : Nat) (std : Term)
    (cur : Nat) (s : State Term) : Prop :=
  (∀ (r : RegionName) o,r≠"out" → s.mem r o=base.mem r o) ∧
  (∀ k,k<p.width → k<cur → s.mem "out" (TensorSource.rowAddress p b m k)=
    .mk .real (storedValue base p b m k std)) ∧
  (∀ (r : RegionName) o,(r≠"out" ∨ ∀ k,k<p.width → o≠TensorSource.rowAddress p b m k) → s.mem r o=base.mem r o)

theorem emit_invariant (base s : State Term) (p : Parameters) (b m B cur : Nat) (std : Term)
    (unique : RowUnique p b m) (hs : WriteInvariant base p b m std cur s) :
    WriteInvariant base p b m std (cur+B) (emit p b m B std cur s) := by
  refine ⟨?_,?_,?_⟩
  · intro r o hr
    exact (emit_frame p b m B std cur s r o (Or.inl hr)).trans (hs.1 r o hr)
  · intro k hk bound
    by_cases lo : cur≤k
    · rw [emit_value p b m B std cur s unique k hk lo bound]
      congr 1
      simp only [storedValue,inputValue,weightValue,if_pos hk,read,
        hs.1 "x" _ (by decide),hs.1 "weights" _ (by decide)]
    · rw [emit_frame p b m B std cur s "out" _ (Or.inr ?_)]
      · exact hs.2.1 k hk (by omega)
      · intro i hi same
        have he := unique _ _ hk hi same
        omega
  · intro r o outside
    apply Eq.trans (emit_frame p b m B std cur s r o ?_) (hs.2.2 r o outside)
    rcases outside with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i hi=>ho _ hi)

theorem graph_spec (s : State Term) (p : Parameters) (b m B : Nat)
    (hp0 : s.pids 0=b) (hp1 : s.pids 1=m) (positive : 0<B) (unique : RowUnique p b m) :
    (∀ k,k<p.width → (graph s p B).mem "out" (TensorSource.rowAddress p b m k)=
      .mk .real (logical p.width B p.eps
        (fun j=>read s "x" (inputAddress p b m j))
        (fun j=>read s "weights" (j*p.weights.stride 0)) k)) ∧
    (∀ (r : RegionName) o,(r≠"out" ∨ ∀ k,k<p.width → o≠TensorSource.rowAddress p b m k) →
      (graph s p B).mem r o=s.mem r o) := by
  let v := (walk (accumulate s p b m B) p.width B 0 (fun _=>lit 0)).2
  let std := standard p B v
  have initial : WriteInvariant s p b m std 0 s := by
    refine ⟨fun _ _ _=>rfl,?_,fun _ _ _=>rfl⟩
    intro k _ hk
    omega
  obtain ⟨done,inv⟩ := walk_invariant (emit p b m B std) p.width B positive
    (WriteInvariant s p b m std) (fun cur t _ ht=>emit_invariant s t p b m B cur std unique ht) 0 s initial
  have he : accumulate s p b m B=laneStep p.width B (fun j=>read s "x" (inputAddress p b m j)) := rfl
  constructor
  · intro k hk
    have hv := inv.2.1 k hk (by omega)
    simpa only [graph,hp0,hp1,storedValue,inputValue,weightValue,if_pos hk,
      standard,reduced,logical,logicalStd,he,v,std] using hv
  · simpa only [graph,hp0,hp1,v,std] using inv.2.2

#axiomsClean graph_spec
#guard_msgs (drop info) in
#specNonCircular logical avoiding [code,kernel,graph,State]
end
end VeriTile.Bench.Repairs.RmsnormTriton.Symbolic

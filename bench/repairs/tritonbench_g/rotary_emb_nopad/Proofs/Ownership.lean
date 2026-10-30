import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Geometry

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Checked

def destination (r : Request) (indices : Indices) (w : Work) (odd : Bool) :=
  r.destination indices w.key w.token w.head (r.dim w.pair odd)

theorem ranges_apart (c : Call) (indices : Indices) (accepted : check c indices=true)
    (i j : Fin 7) (pi : present c.request i=true) (pj : present c.request j=true) (different : i≠j)
    (a b : Nat) (ha : base c.request i≤a ∧ a<base c.request i+wordExtent c.request i)
    (hb : base c.request j≤b ∧ b<base c.request j+wordExtent c.request j) : a≠b := by
  have ni : 0<wordExtent c.request i := by omega
  have nj : 0<wordExtent c.request j := by omega
  rcases physical_separate c indices accepted i j pi pj different ni nj with first|second <;> omega

theorem destination_role_injective (r : Request) (a b : Bool)
    (same : destinationRole r a=destinationRole r b) : a=b := by
  cases hc : r.cached <;> cases a <;> cases b <;> simp_all [destinationRole,role]

theorem destination_injective (c : Call) (indices : Indices) (accepted : check c indices=true)
    (a b : Work) (ha : c.request.Active a) (hb : c.request.Active b) (oa ob : Bool)
    (same : destination c.request indices a oa=destination c.request indices b ob) : a=b ∧ oa=ob := by
  have roles : destinationRole c.request a.key=destinationRole c.request b.key := by
    by_contra different
    exact ranges_apart c indices accepted _ _ (destination_present _ _) (destination_present _ _)
      different _ _ (destination_range c indices accepted a ha oa) (destination_range c indices accepted b hb ob) same
  have keys := destination_role_injective c.request a.key b.key roles
  have dimA := dim_bound c.request a ha oa
  have dimB := dim_bound c.request b hb ob
  by_cases cached : (c.request.cached && a.key)=true
  · have flags : c.request.cached=true ∧ a.key=true := by simpa only [Bool.and_eq_true] using cached
    have bk : b.key=true := keys.symm.trans flags.2
    have slotsA := slots c indices accepted a ha flags.1 a.token ha.1
    have slotsB := slots c indices accepted b hb flags.1 b.token hb.1
    have headA : a.head<c.request.kHeads := by simpa [Request.heads,flags.2] using ha.2.1
    have headB : b.head<c.request.kHeads := by simpa [Request.heads,bk] using hb.2.1
    have eq : ((c.request.block indices a.token*c.request.kHeads+a.head)*c.request.blockSize+
        c.request.slot indices a.token)*c.request.width+c.request.dim a.pair oa=
      ((c.request.block indices b.token*c.request.kHeads+b.head)*c.request.blockSize+
        c.request.slot indices b.token)*c.request.width+c.request.dim b.pair ob := by
      simpa [destination,Request.destination,flags.1,flags.2,bk,Nat.add_assoc] using same
    have dims := mixed_injective c.request.width _ _ _ _ dimA dimB eq
    have rows := mixed_injective c.request.blockSize _ _ _ _ slotsA.2 slotsB.2 dims.1
    have heads := mixed_injective c.request.kHeads _ _ _ _ headA headB rows.1
    have token := slots_unique c indices accepted a ha flags.1 a.token b.token ha.1 hb.1 (Prod.ext heads.1 rows.2)
    have pair := dim_injective c.request a b ha hb oa ob dims.2
    exact ⟨by cases a; cases b; simp_all,pair.2⟩
  · have cachedB : ¬(c.request.cached && b.key)=true := by simpa [keys] using cached
    have eq : (a.token*c.request.heads a.key+a.head)*c.request.width+c.request.dim a.pair oa=
        (b.token*c.request.heads a.key+b.head)*c.request.width+c.request.dim b.pair ob := by
      simpa [destination,Request.destination,cached,cachedB,Request.data,←keys,Nat.add_assoc] using same
    have dims := mixed_injective c.request.width _ _ _ _ dimA dimB eq
    have headB : b.head<c.request.heads a.key := by simpa [keys] using hb.2.1
    have rows := mixed_injective (c.request.heads a.key) _ _ _ _ ha.2.1 headB dims.1
    have pair := dim_injective c.request a b ha hb oa ob dims.2
    exact ⟨by cases a; cases b; simp_all,pair.2⟩

theorem different_destinations (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) : destination c.request indices w false≠destination c.request indices w true := by
  intro same
  have h := (destination_injective c indices accepted w w active active false true same).2
  cases h

theorem destination_input_key (r : Request) (a b : Bool) (same : destinationRole r a=role b) :
    a=b ∧ (r.cached && b)=false := by
  cases hc : r.cached <;> cases a <;> cases b <;> simp_all [destinationRole,role]

theorem other_input_preserved (c : Call) (indices : Indices) (accepted : check c indices=true)
    (a b : Work) (ha : c.request.Active a) (hb : c.request.Active b) (different : a≠b) (oa ob : Bool) :
    destination c.request indices a oa≠c.request.data b.key b.token b.head (c.request.dim b.pair ob) := by
  by_cases roles : destinationRole c.request a.key=role b.key
  · obtain ⟨_,notCache⟩ := destination_input_key c.request a.key b.key roles
    intro same
    have eq : destination c.request indices a oa=destination c.request indices b ob := by
      simpa [destination,Request.destination,notCache] using same
    exact different (destination_injective c indices accepted a b ha hb oa ob eq).1
  · apply ranges_apart c indices accepted _ _ (destination_present _ _) _ roles _ _
      (destination_range c indices accepted a ha oa) (input_range c.request b hb ob)
    cases b.key <;> simp [present,role]

theorem coefficient_preserved (c : Call) (indices : Indices) (accepted : check c indices=true)
    (a b : Work) (ha : c.request.Active a) (hb : c.request.Active b) (odd sine : Bool) :
    destination c.request indices a odd≠c.request.coefficient sine b.token b.pair := by
  let i : Fin 7 := if sine then 3 else 2
  have pi : present c.request i=true := by cases sine <;> simp [i,present]
  have different : destinationRole c.request a.key≠i := by
    cases hc : c.request.cached <;> cases hk : a.key <;> cases sine <;> simp [destinationRole,role,hc,i]
  exact ranges_apart c indices accepted _ i (destination_present _ _) pi different _ _
    (destination_range c indices accepted a ha odd) (coefficient_range c indices accepted b hb sine)

/-- Entire read-only role intervals are outside the write set. In cache mode
this includes the original K allocation as well as coefficients and metadata. -/
theorem readonly_role (c : Call) (indices : Indices) (accepted : check c indices=true)
    (i : Fin 7) (pi : present c.request i=true)
    (readonly : ∀ key : Bool,destinationRole c.request key≠i)
    (w : Work) (active : c.request.Active w) (odd : Bool) (offset : Nat)
    (inside : offset<wordExtent c.request i) :
    destination c.request indices w odd≠base c.request i+offset := by
  exact ranges_apart c indices accepted _ i (destination_present _ _) pi (readonly w.key) _ _
    (destination_range c indices accepted w active odd) ⟨by omega,by omega⟩

#axiomsClean destination_injective
#axiomsClean other_input_preserved
#axiomsClean coefficient_preserved
#axiomsClean readonly_role
end VeriTile.Bench.Repairs.RotaryEmbNopad.Checked

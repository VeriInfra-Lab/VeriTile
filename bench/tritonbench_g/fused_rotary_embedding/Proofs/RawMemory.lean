import bench.tritonbench_g.fused_rotary_embedding.Proofs.Connection

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Read projection only. Exterior cells are retained by `rawCommit`, not
replaced by a normalized memory. -/
def rawValues (m : Mem α) : Memory α := fun r a => (m (region r) a).read .real

def rawStore (m : Mem α) (w : Stores.Write α) : Mem α :=
  fun r a => if r = region w.1.1 ∧ a = w.1.2 then .mk .real w.2 else m r a

def rawCommit (ws : List (Stores.Write α)) (m : Mem α) : Mem α :=
  ws.foldl rawStore m

@[simp] theorem region_output_eq (r : FusedRotaryEmbedding.Region) (kind : Kind) :
    region r = region kind.output ↔ r = kind.output := by
  cases r <;> cases kind <;> simp [region,Kind.output]

theorem rawStore_values (m : Mem α) (kind : Kind) (a : Nat) (v : α) :
    rawValues (rawStore m ((kind.output,a),v)) =
      Stores.store (rawValues m) ((kind.output,a),v) := by
  funext r off
  simp only [rawValues, rawStore, Stores.store, Prod.mk.injEq, region_output_eq]
  split_ifs <;> simp [Cell.read]

theorem rawCommit_values (ws : List (Stores.Write α))
    (outputs : ∀ w ∈ ws, ∃ kind : Kind, w.1.1 = kind.output) (m : Mem α) :
    rawValues (rawCommit ws m) = Stores.commit ws (rawValues m) := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    obtain ⟨kind, hk⟩ := outputs w (by simp)
    rcases w with ⟨⟨r,a⟩,v⟩
    dsimp at hk
    subst r
    simp only [rawCommit, Stores.commit, List.foldl_cons] at *
    rw [ih (fun w hw => outputs w (by simp [hw])), rawStore_values]

theorem rawCommit_frame (ws : List (Stores.Write α)) (m : Mem α) (r : RegionName) (a : Nat)
    (outside : ∀ w ∈ ws, ¬(r = region w.1.1 ∧ a = w.1.2)) :
    rawCommit ws m r a = m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change rawCommit ws (rawStore m w) r a = m r a
    rw [ih _ (fun x hx => outside x (by simp [hx]))]
    simp [rawStore, outside w (by simp)]

theorem rawCommit_real (ws : List (Stores.Write α)) (m : Mem α) (r : RegionName) (a : Nat)
    (already : ∃ x, m r a = .mk .real x) :
    ∃ x, rawCommit ws m r a = .mk .real x := by
  induction ws generalizing m with
  | nil => exact already
  | cons w ws ih =>
    apply ih (rawStore m w)
    by_cases hit : r = region w.1.1 ∧ a = w.1.2
    · exact ⟨w.2, by simp [rawStore,hit]⟩
    · simpa [rawStore,hit] using already

theorem rawCommit_written (ws : List (Stores.Write α)) (m : Mem α) (r : RegionName) (a : Nat)
    (written : ∃ w ∈ ws, r = region w.1.1 ∧ a = w.1.2) :
    ∃ x, rawCommit ws m r a = .mk .real x := by
  induction ws generalizing m with
  | nil => simpa using written
  | cons w ws ih =>
    rcases written with ⟨v,hv,hit⟩
    rcases List.mem_cons.mp hv with eq | hv
    · subst v
      exact rawCommit_real ws (rawStore m w) r a ⟨w.2, by simp [rawStore,hit]⟩
    · exact ih (rawStore m w) ⟨v,hv,hit⟩

/-- Precisely the indirect metadata reads performed by live programs. -/
def MetadataBinding (c : Config) (m : Mem α) : Prop :=
  (∀ t, t < c.shape.tokens → (m "context_lengths" t).read .int = (c.metadata.lengths t : Int)) ∧
  (∀ t, t < c.shape.tokens →
    (m "BLOCK_TABLES" (t*c.shape.tableWidth+c.past t/c.shape.blockSize)).read .int = (c.block t : Int))

theorem rawCommit_metadata (c : Config) (ws : List (Stores.Write α))
    (outputs : ∀ w ∈ ws, ∃ kind : Kind, w.1.1 = kind.output) (m : Mem α)
    (hm : MetadataBinding c m) : MetadataBinding c (rawCommit ws m) := by
  have untouched (r : RegionName) (hr : r = "context_lengths" ∨ r = "BLOCK_TABLES") (a : Nat) :
      rawCommit ws m r a = m r a := by
    apply rawCommit_frame
    intro w hw hit
    obtain ⟨kind,hk⟩ := outputs w hw
    rw [hk] at hit
    cases kind <;> rcases hr with rfl | rfl <;> simp [region,Kind.output] at hit
  constructor
  · intro t ht
    rw [untouched _ (Or.inl rfl)]
    exact hm.1 t ht
  · intro t ht
    rw [untouched _ (Or.inr rfl)]
    exact hm.2 t ht

@[simp] theorem raw_past (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2 < c.shape.tokens) (m : Mem α) (hm : MetadataBinding c m) :
    past (parameters c.shape) (state pid m) = (c.past pid.2 : Int) := by
  have positive := hc.lengthPositive pid.2 ht
  simp only [past,parameters,state,Nat.cast_one,Int.mul_one,Int.toNat_natCast,
    Nat.one_ne_zero,ite_false,ite_true,hm.1 _ ht]
  unfold Config.past
  omega

@[simp] theorem raw_block (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2 < c.shape.tokens) (m : Mem α) (hm : MetadataBinding c m) :
    block (parameters c.shape) (state pid m) = (c.block pid.2 : Int) := by
  unfold block
  rw [raw_past c hc pid ht m hm]
  simp only [parameters,quot_nat,state,Nat.cast_one,Int.mul_one,
    ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  exact hm.2 _ ht

@[simp] theorem raw_slot (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2 < c.shape.tokens) (m : Mem α) (hm : MetadataBinding c m) :
    slot (parameters c.shape) (state pid m) = (c.slot pid.2 : Int) := by
  unfold slot
  rw [raw_past c hc pid ht m hm]
  exact rem_nat _ _

theorem raw_cacheOffset (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2 < c.shape.tokens) (m : Mem α) (hm : MetadataBinding c m)
    (high : Bool) (i : HalfLane (parameters c.shape)) (hi : i.1.val < c.shape.half) :
    cacheOffset (parameters c.shape) (state pid m) high i =
      c.output (work c.shape pid Bool.true i.1.val) high := by
  have dimBound : i.1.val+(if high then c.shape.half else 0)<c.shape.width := by
    have := width_halves c.shape hc.shape
    cases high <;> simp <;> omega
  simp only [cacheOffset,raw_block c hc pid ht m hm,head_nat,
    raw_slot c hc pid ht m hm,dimension_nat]
  cases layout : c.shape.newLayout <;>
    simp only [parameters,layout,Bool.false_eq_true,ite_false,ite_true,
      quot_nat,rem_nat,Config.output,work,Shape.cache,Shape.dim,Kind.noConfusion,
      Bool.and_self,Nat.cast_zero,Nat.cast_one,Int.mul_zero,Int.mul_one,Int.add_zero,
      Nat.mod_eq_of_lt dimBound,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast] <;> simp <;> ring

theorem raw_valueCacheOffset (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2 < c.shape.tokens) (m : Mem α) (hm : MetadataBinding c m)
    (i : FullLane (parameters c.shape)) :
    valueCacheOffset (parameters c.shape) (state pid m) i =
      c.shape.cache Bool.false (c.block pid.2) (pid.1/c.shape.groups) (c.slot pid.2) i.1.val := by
  simp only [valueCacheOffset,raw_block c hc pid ht m hm,head_nat,raw_slot c hc pid ht m hm]
  simp only [parameters,Shape.cache,Bool.false_and,Bool.false_eq_true,ite_false,ite_true,
    Nat.cast_one,Int.mul_one,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  simp
  ring

theorem raw_pair (rest : Algebra α) (ops : Ops α) (c : Config)
    (pid : Program) (m : Mem α) (key high : Bool) (i : HalfLane (parameters c.shape))
    (hi : i.1.val < c.shape.half) :
    pair rest ops (parameters c.shape) (state pid m) key high i =
      .mk .real (component (pairValues c.shape (fun _=>ops) (work c.shape pid key i.1.val)
        (rawValues m)) high) := by
  simp only [pair,load,active_nat,hi,ite_true,dataOffset_nat,coefficientOffset_nat]
  cases key <;> simp [state,rawValues,region,pairValues,work,Kind.input]

theorem rawCommit_state {ι : Type} (xs : List ι) (kind : Kind)
    (off : ι → Nat) (value : ι → α) (s : State α) :
    (xs.foldl (fun t i => t.write (region kind.output) (off i) (.mk .real (value i))) s).mem =
      rawCommit (xs.map (fun i => ((kind.output,off i),value i))) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons,List.map_cons,rawCommit] at *
    rw [ih]
    rfl

theorem raw_scatter (kind : Kind) (n limit : Nat) (bound : limit ≤ n)
    (mask : TileIndex [n] → Prop) [DecidablePred mask]
    (off : TileIndex [n] → Nat) (value : TileIndex [n] → Cell α)
    (naturalOff : Nat → Nat) (naturalValue : Nat → α)
    (masks : ∀ i, mask i ↔ i.1.val < limit)
    (offsets : ∀ i, mask i → off i = naturalOff i.1.val)
    (values : ∀ i, mask i → value i = .mk .real (naturalValue i.1.val)) (s : State α) :
    (scatter (TileShape.allIndices [n]) (region kind.output) mask off value s).mem =
      rawCommit ((List.range limit).map (fun i => ((kind.output,naturalOff i),naturalValue i))) s.mem := by
  rw [scatter_congr _ _ _ off (fun i=>naturalOff i.1.val) value
    (fun i=>.mk .real (naturalValue i.1.val)) (fun i _=>offsets i) (fun i _=>values i)]
  unfold scatter
  rw [State.masked_scatter_eq_filter,rawCommit_state]
  have masks' : (fun i : TileIndex [n]=>decide (mask i))=(fun i=>decide (i.1.val<limit)) := by
    funext i
    simp [masks i]
  rw [masks',active_lanes n limit bound (fun i=>((kind.output,naturalOff i),naturalValue i))]

#axiomsClean rawCommit_values
#axiomsClean rawCommit_written
#axiomsClean raw_scatter
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source

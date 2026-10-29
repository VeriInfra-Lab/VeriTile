import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeWindowHits

/-! Address ownership for unrestricted original native configurations. In
particular no positive BT or BS>=16 premise is used to justify serialization. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem inner_job_bound (c : HostFP.Inner) (j : Job c.plans) : number j<c.total := by
  have member := plan_mem j
  have valid : LaunchPlan.Valid c.total (plan j) := by
    unfold HostFP.Inner.plans at member
    split at member
    · simp at member
    · exact LaunchPlan.plans_valid _ _ member
  have lane := j.2.isLt
  change j.2.val<(plan j).size at lane
  change (plan j).base+j.2.val<c.total
  have ending := valid.2.2
  omega

theorem axis_bound (size block n : Nat) (positive : 0<size/block) (i : Fin block) :
    (n%(size/block))*block+i.val<size := by
  have tile := Nat.mul_le_mul_right block (Nat.succ_le_iff.mpr (Nat.mod_lt n positive))
  have capacity := Nat.div_mul_le_self size block
  have lane := i.isLt
  simp only [Nat.succ_mul] at tile
  omega

/-- Equality of row-major addresses recovers both decoded tile coordinates
and the arbitrary outer row, using only actual bounded tile lanes. -/
theorem tile_position (K V BK BV n m outerN outerM : Nat)
    (hx : 0<V/BV) (hy : 0<K/BK) (e f : Fin BK) (i j : Fin BV)
    (equal : (outerN*K+(Grid.decodeY (V/BV) (K/BK) n*BK+e.val))*V+
        (Grid.decodeX (V/BV) n*BV+i.val) =
      (outerM*K+(Grid.decodeY (V/BV) (K/BK) m*BK+f.val))*V+
        (Grid.decodeX (V/BV) m*BV+j.val)) :
    outerN=outerM ∧ Grid.decodeY (V/BV) (K/BK) n=Grid.decodeY (V/BV) (K/BK) m ∧
      Grid.decodeX (V/BV) n=Grid.decodeX (V/BV) m := by
  have vn := axis_bound V BV n hx i
  have vm := axis_bound V BV m hx j
  have kn := axis_bound K BK (n/(V/BV)) hy e
  have km := axis_bound K BK (m/(V/BV)) hy f
  have v := Grid.batch_no_duplicate (Nat.zero_lt_of_lt vn) vn vm equal
  have k := Grid.batch_no_duplicate (Nat.zero_lt_of_lt kn) kn km v.1
  have kb := Grid.batch_no_duplicate (Nat.zero_lt_of_lt e.isLt) e.isLt f.isLt k.2
  have vb := Grid.batch_no_duplicate (Nat.zero_lt_of_lt i.isLt) i.isLt j.isLt v.2
  exact ⟨k.1,kb.1,vb.1⟩

theorem decode_unique (nx ny n m : Nat) (hx : 0<nx) (hy : 0<ny)
    (head : Grid.decodeZ nx ny n=Grid.decodeZ nx ny m)
    (row : Grid.decodeY nx ny n=Grid.decodeY nx ny m)
    (col : Grid.decodeX nx n=Grid.decodeX nx m) : n=m := by
  calc
    n=Grid.encode nx ny (Grid.decodeX nx n) (Grid.decodeY nx ny n) (Grid.decodeZ nx ny n) :=
      (Grid.decode_encode_identity hx hy).symm
    _=Grid.encode nx ny (Grid.decodeX nx m) (Grid.decodeY nx ny m) (Grid.decodeZ nx ny m) := by
      rw [head,row,col]
    _=m := Grid.decode_encode_identity hx hy

theorem inner_owns_hits (c : HostFP.Inner) (p : LaunchPlan.Item) (lane : Fin p.size)
    (s : State NativeGraph.Expr) (r : RegionName) (off : Nat) :
    NativeGrid.innerOwns c p lane r off ↔ NativeStores.Hits
      (NativeStores.inner (NativeGrid.innerShape c) p.base (HostFP.gridState s p.size lane.val)
        c.gateK c.hasSeed c.storeFinal) r off := by
  rw [NativeStores.inner_hits]
  simp only [NativeGrid.innerOwns,HostFP.Inner.owns]
  constructor
  · rintro ⟨e,he,f,hf,(⟨hr,n,hn,ho⟩|⟨enabled,hr,ho⟩)⟩
    · refine Or.inl ⟨n,hn,hr,(⟨e,he⟩,⟨f,hf⟩,PUnit.unit),?_⟩
      rw [ho]
      simp only [NativeValue.historyOffset,NativeValue.head,NativeValue.kblock,NativeValue.vblock,
        NativeValue.linear,NativeGrid.innerShape,HostFP.gridState,↓reduceIte,
        HostFP.Inner.NT,HostFP.Inner.PT,HostFP.Inner.PK,HostFP.Inner.PV,
        HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,
        Grid.decodeX,Grid.decodeY,Grid.decodeZ,Nat.add_comm lane.val p.base]
      ring
    · refine Or.inr ⟨enabled,hr,(⟨e,he⟩,⟨f,hf⟩,PUnit.unit),?_⟩
      rw [ho]
      simp only [NativeValue.stateOffset,NativeValue.head,NativeValue.kblock,NativeValue.vblock,
        NativeValue.linear,NativeGrid.innerShape,HostFP.gridState,↓reduceIte,
        HostFP.Inner.NT,HostFP.Inner.PT,HostFP.Inner.PK,HostFP.Inner.PV,
        HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,
        Grid.decodeX,Grid.decodeY,Grid.decodeZ,Nat.add_comm lane.val p.base]
      ring
  · rintro (⟨n,hn,hr,i,ho⟩|⟨enabled,hr,i,ho⟩)
    · refine ⟨i.1.val,i.1.isLt,i.2.1.val,i.2.1.isLt,Or.inl ⟨hr,n,hn,?_⟩⟩
      rw [ho]
      simp only [NativeValue.historyOffset,NativeValue.head,NativeValue.kblock,NativeValue.vblock,
        NativeValue.linear,NativeGrid.innerShape,HostFP.gridState,↓reduceIte,
        HostFP.Inner.NT,HostFP.Inner.PT,HostFP.Inner.PK,HostFP.Inner.PV,
        HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,
        Grid.decodeX,Grid.decodeY,Grid.decodeZ,Nat.add_comm lane.val p.base]
      ring
    · refine ⟨i.1.val,i.1.isLt,i.2.1.val,i.2.1.isLt,Or.inr ⟨enabled,hr,?_⟩⟩
      rw [ho]
      simp only [NativeValue.stateOffset,NativeValue.head,NativeValue.kblock,NativeValue.vblock,
        NativeValue.linear,NativeGrid.innerShape,HostFP.gridState,↓reduceIte,
        HostFP.Inner.NT,HostFP.Inner.PT,HostFP.Inner.PK,HostFP.Inner.PV,
        HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,
        Grid.decodeX,Grid.decodeY,Grid.decodeZ,Nat.add_comm lane.val p.base]
      ring

theorem prefix_owns_hits (c : HostFP.Cumulative) (p : LaunchPlan.Item) (lane : Fin p.size)
    (s : State NativeGraph.Expr) (r : RegionName) (off : Nat) :
    NativeGrid.prefixOwns c p lane r off ↔ NativeStores.Hits
      (NativeStores.prefixStores (NativeGrid.prefixShape c) p.base (HostFP.gridState s p.size lane.val)) r off := by
  rw [NativeStores.prefix_hits]
  simp only [NativeGrid.prefixOwns,HostFP.Cumulative.owns]
  constructor
  · rintro ⟨active,hr,t,ht,f,hf,ho⟩
    refine ⟨?_,hr,(⟨t,ht⟩,⟨f,hf⟩,PUnit.unit),?_⟩
    · simpa only [NativePrefix.enabled,NativePrefix.linear,HostFP.gridState,NativeGrid.prefixShape,
        ↓reduceIte,HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,Nat.add_comm lane.val p.base] using active
    · rw [ho]
      simp only [NativePrefix.offset,NativePrefix.head,NativePrefix.timeBlock,NativePrefix.featureBlock,
        NativePrefix.linear,NativeGrid.prefixShape,HostFP.gridState,↓reduceIte,
        HostFP.Inner.NT,HostFP.Inner.PT,HostFP.Inner.PK,HostFP.Inner.PV,
        HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,
        Grid.decodeX,Grid.decodeY,Grid.decodeZ,Nat.add_comm lane.val p.base]
      ring
  · rintro ⟨active,hr,i,ho⟩
    refine ⟨?_,hr,i.1.val,i.1.isLt,i.2.1.val,i.2.1.isLt,?_⟩
    · simpa only [NativePrefix.enabled,NativePrefix.linear,HostFP.gridState,NativeGrid.prefixShape,
        ↓reduceIte,HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,Nat.add_comm lane.val p.base] using active
    · rw [ho]
      simp only [NativePrefix.offset,NativePrefix.head,NativePrefix.timeBlock,NativePrefix.featureBlock,
        NativePrefix.linear,NativeGrid.prefixShape,HostFP.gridState,↓reduceIte,
        HostFP.Inner.NT,HostFP.Inner.PT,HostFP.Inner.PK,HostFP.Inner.PV,
        HostFP.Cumulative.PT,HostFP.Cumulative.PS,HostFP.Cumulative.total,
        Grid.decodeX,Grid.decodeY,Grid.decodeZ,Nat.add_comm lane.val p.base]
      ring

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface

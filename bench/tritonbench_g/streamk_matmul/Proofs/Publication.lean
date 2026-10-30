import bench.tritonbench_g.streamk_matmul.Proofs.Source
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Source
open Protocol Plan
set_option maxHeartbeats 12000000
noncomputable section

@[simp] theorem lane_mem (p : Parameters) (lane : Lane p) : lane ∈ lanes p := by
  simp [lanes]

theorem scratch_address_unique (p : Parameters) (pr qr : Fin p.h.programs) (l k : Lane p)
    (he : scratch p pr l=scratch p qr k) : pr=qr ∧ l=k := by
  have hv:=congrArg Fin.val he
  have hu:=@Geometry.scratch_injective p.h.programs p.h.bm p.h.bn (pr,l) (qr,k) hv
  have hp:pr=qr:=congrArg (fun z : Fin p.h.programs × Fin p.h.bm × Fin p.h.bn => z.1) hu
  have hl:l=k:=congrArg (fun z : Fin p.h.programs × Fin p.h.bm × Fin p.h.bn => z.2) hu
  exact ⟨hp,hl⟩

theorem storeTile_no_scratch (p : Parameters)
    (value : Lane p → (A p → α) → (B p → α) → α) (tile : Nat) (e : Event α p)
    (he : e ∈ storeTile p value tile) (i : S p) : ¬Memory.WritesScratch i e := by
  unfold storeTile at he
  obtain ⟨lane,hl,he⟩:=List.mem_flatMap.mp he
  split at he
  next h => simp only [List.mem_singleton] at he; subst e; simp [Memory.WritesScratch]
  next h => simp at he

theorem first_scratch_member (p : Parameters) (ops : Instructions α)
    (hs : 0 < (select p.h).streamTiles) (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val
      (iterations_positive p.h (selected_active p.h hs).2 p.blockK))
    (hn : c.hi%(select p.h).iterations≠0) (lane : Lane p) :
    Memory.Event.storeScratch (scratch p pr lane)
      (fun a b=>chunkValue p ops a b (c.tile (select p.h).iterations) c.lo c.hi lane) ∈ firstWave p ops := by
  unfold firstWave
  rw [dif_pos hs]
  apply List.mem_flatMap.mpr
  refine ⟨pr,by simp,?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨c,hc,?_⟩
  simp only [macLoop,if_neg hn]
  exact List.mem_map.mpr ⟨lane,lane_mem p lane,rfl⟩

/-- Every actual first-wave event touching this scratch address is its one
proven source writer, with the same explicit local chunk expression. -/
theorem first_scratch_unique (p : Parameters) (ops : Instructions α)
    (hs : 0 < (select p.h).streamTiles) (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val
      (iterations_positive p.h (selected_active p.h hs).2 p.blockK))
    (hn : c.hi%(select p.h).iterations≠0) (lane : Lane p)
    (e : Event α p) (he : e ∈ firstWave p ops) (hw : Memory.WritesScratch (scratch p pr lane) e) :
    e=Memory.Event.storeScratch (scratch p pr lane)
      (fun a b=>chunkValue p ops a b (c.tile (select p.h).iterations) c.lo c.hi lane) := by
  unfold firstWave at he
  rw [dif_pos hs] at he
  obtain ⟨qr,hqr,he⟩:=List.mem_flatMap.mp he
  obtain ⟨d,hd,he⟩:=List.mem_flatMap.mp he
  unfold macLoop at he
  dsimp only at he
  split at he
  next ht => exact False.elim (storeTile_no_scratch p _ _ e he _ hw)
  next ht =>
    obtain ⟨other,ho,he⟩:=List.mem_map.mp he
    subst e
    change scratch p qr other=scratch p pr lane at hw
    obtain ⟨hp,hl⟩:=scratch_address_unique p qr pr other lane hw
    subst qr
    subst other
    have dc:=scratch_unique (select p.h).iterations _ _ _ d c hd hc ht hn
    subst d
    rfl

/-- Publication is an actual finite-memory equation after first_wave, not an
assumed visibility or output equation. The next launch reads this value. -/
theorem first_scratch_value (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (hs : 0 < (select p.h).streamTiles) (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val
      (iterations_positive p.h (selected_active p.h hs).2 p.blockK))
    (hn : c.hi%(select p.h).iterations≠0) (lane : Lane p) :
    (Memory.run ops.atomicAdd (firstWave p ops) initial).scratch (scratch p pr lane)=
      chunkValue p ops initial.inputA initial.inputB (c.tile (select p.h).iterations) c.lo c.hi lane := by
  apply Memory.scratch_writer_result ops.atomicAdd (firstWave p ops) initial (scratch p pr lane)
    (fun a b=>chunkValue p ops a b (c.tile (select p.h).iterations) c.lo c.hi lane)
  · exact first_scratch_member p ops hs pr c hc hn lane
  · exact first_scratch_unique p ops hs pr c hc hn lane

/-- All first-wave chunk tiles are in the host stream prefix. -/
theorem chunk_tile_bound (p : Parameters) (hs : 0 < (select p.h).streamTiles)
    (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val
      (iterations_positive p.h (selected_active p.h hs).2 p.blockK)) :
    c.tile (select p.h).iterations < (select p.h).streamTiles := by
  have hI:=iterations_positive p.h (selected_active p.h hs).2 p.blockK
  have hP:=(selected_active p.h hs).1
  have shape:=member_shape _ _ _ hI c hc
  have hlast:=edge_last ((select p.h).streamTiles*(select p.h).iterations) p.h.programs hP
  have hm:=edge_mono ((select p.h).streamTiles*(select p.h).iterations/p.h.programs)
    ((select p.h).streamTiles*(select p.h).iterations%p.h.programs)
    (pr.val+1) p.h.programs (by omega)
  apply (Nat.div_lt_iff_lt_mul hI).mpr
  omega


theorem output_address_unique (p : Parameters) (tile other : Nat)
    (ht : tile < tiles p.h) (ho : other < tiles p.h) (l k : Lane p)
    (hl : active p tile l) (hk : active p other k)
    (he : output p tile l hl=output p other k hk) : tile=other ∧ l=k := by
  let x : Fin p.h.m × Fin p.h.n := (⟨row p tile l,hl.1⟩,⟨col p tile l,hl.2⟩)
  let y : Fin p.h.m × Fin p.h.n := (⟨row p other k,hk.1⟩,⟨col p other k,hk.2⟩)
  have hxy:=@Geometry.packed_injective p.h.m p.h.n x y (congrArg Fin.val he)
  have hr:row p tile l=row p other k:=congrArg (fun z : Fin p.h.m × Fin p.h.n => z.1.val) hxy
  have hn:col p tile l=col p other k:=congrArg (fun z : Fin p.h.m × Fin p.h.n => z.2.val) hxy
  have htile:tile=other:=by
    apply IntScaledMatmul.Revision001.Geometry.owner_unique (Geometry.layout p.h)
      (Geometry.valid p.h p.blockM p.blockN) tile other ht ho x
    · exact ⟨(l.1,l.2,PUnit.unit),rfl,rfl⟩
    · exact ⟨(k.1,k.2,PUnit.unit),hr.symm,hn.symm⟩
  refine ⟨htile,?_⟩
  subst other
  apply Prod.ext
  · apply Fin.ext
    simp only [row] at hr
    omega
  · apply Fin.ext
    simp only [col] at hn
    omega

theorem first_output_member (p : Parameters) (ops : Instructions α)
    (hs : 0 < (select p.h).streamTiles) (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val
      (iterations_positive p.h (selected_active p.h hs).2 p.blockK))
    (ht : c.hi%(select p.h).iterations=0) (lane : Lane p)
    (hl : active p (c.tile (select p.h).iterations) lane) :
    Memory.Event.storeOutput (output p (c.tile (select p.h).iterations) lane hl)
      (fun a b=>chunkValue p ops a b (c.tile (select p.h).iterations) c.lo c.hi lane) ∈ firstWave p ops := by
  unfold firstWave
  rw [dif_pos hs]
  apply List.mem_flatMap.mpr
  refine ⟨pr,by simp,?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨c,hc,?_⟩
  simp only [macLoop,if_pos ht,storeTile]
  apply List.mem_flatMap.mpr
  refine ⟨lane,lane_mem p lane,?_⟩
  rw [dif_pos hl]
  exact List.mem_singleton_self _

theorem first_output_unique (p : Parameters) (ops : Instructions α)
    (hs : 0 < (select p.h).streamTiles) (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val
      (iterations_positive p.h (selected_active p.h hs).2 p.blockK))
    (ht : c.hi%(select p.h).iterations=0) (lane : Lane p)
    (hl : active p (c.tile (select p.h).iterations) lane)
    (e : Event α p) (he : e ∈ firstWave p ops)
    (hw : Memory.WritesOutput (output p (c.tile (select p.h).iterations) lane hl) e) :
    e=Memory.Event.storeOutput (output p (c.tile (select p.h).iterations) lane hl)
      (fun a b=>chunkValue p ops a b (c.tile (select p.h).iterations) c.lo c.hi lane) := by
  unfold firstWave at he
  rw [dif_pos hs] at he
  obtain ⟨qr,hqr,he⟩:=List.mem_flatMap.mp he
  obtain ⟨d,hd,he⟩:=List.mem_flatMap.mp he
  unfold macLoop at he
  dsimp only at he
  split at he
  next hdterm =>
    unfold storeTile at he
    obtain ⟨other,hother,he⟩:=List.mem_flatMap.mp he
    split at he
    next hactive =>
      simp only [List.mem_singleton] at he
      subst e
      change output p (d.tile (select p.h).iterations) other hactive=
        output p (c.tile (select p.h).iterations) lane hl at hw
      have bd:=lt_of_lt_of_le (chunk_tile_bound p hs qr d hd) (selected_bounded p.h)
      have bc:=lt_of_lt_of_le (chunk_tile_bound p hs pr c hc) (selected_bounded p.h)
      obtain ⟨hteq,heq⟩:=output_address_unique p _ _ bd bc other lane hactive hl hw
      have unique:=initializer_unique (select p.h).streamTiles (select p.h).iterations p.h.programs
        (c.tile (select p.h).iterations) qr.val pr.val _ d c
        ⟨qr.isLt,hd,hteq,hdterm⟩ ⟨pr.isLt,hc,rfl,ht⟩
      have dc:=unique.2
      subst d
      subst other
      rfl
    next hactive => simp at he
  next hdterm =>
    obtain ⟨other,hother,he⟩:=List.mem_map.mp he
    subst e
    simp [Memory.WritesOutput] at hw

/-- The terminal chunk actually initializes every active output lane before
the reduction launch, independent of torch.empty's initial payload. -/
theorem first_output_value (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (hs : 0 < (select p.h).streamTiles) (pr : Fin p.h.programs) (c : Chunk)
    (hc : c ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs pr.val
      (iterations_positive p.h (selected_active p.h hs).2 p.blockK))
    (ht : c.hi%(select p.h).iterations=0) (lane : Lane p)
    (hl : active p (c.tile (select p.h).iterations) lane) :
    (Memory.run ops.atomicAdd (firstWave p ops) initial).output
      (output p (c.tile (select p.h).iterations) lane hl)=
      chunkValue p ops initial.inputA initial.inputB (c.tile (select p.h).iterations) c.lo c.hi lane := by
  apply Memory.output_writer_result ops.atomicAdd (firstWave p ops) initial
    (output p (c.tile (select p.h).iterations) lane hl)
    (fun a b=>chunkValue p ops a b (c.tile (select p.h).iterations) c.lo c.hi lane)
  · exact first_output_member p ops hs pr c hc ht lane hl
  · exact first_output_unique p ops hs pr c hc ht lane hl

#axiomsClean output_address_unique
#axiomsClean first_output_member
#axiomsClean first_output_unique
#axiomsClean first_output_value

#axiomsClean first_scratch_member
#axiomsClean first_scratch_unique
#axiomsClean first_scratch_value
#axiomsClean chunk_tile_bound
end
end Source
end VeriTile.Bench.Repairs.StreamK.Whole001

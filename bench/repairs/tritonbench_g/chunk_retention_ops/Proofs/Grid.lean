import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Execution
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.AddressSafety
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Source

theorem tile_tag (tag : Region) (h s rb cb nr nc rows cols : Nat) (a : Address)
    (ha : a∈tile tag h s rb cb nr nc rows cols) : a.region=tag := by
  simp only [tile,List.mem_flatMap,List.mem_range,List.mem_filterMap] at ha
  rcases ha with ⟨i,hi,j,hj,eq⟩
  split_ifs at eq with hab
  · cases Option.some.inj eq
    rfl

theorem tile_member (tag : Region) (h s rb cb nr nc rows cols i j : Nat)
    (hi : i<nr) (hj : j<nc) (hr : rb+i<rows) (hc : cb+j<cols) :
    (⟨tag,h,s,rb+i,cb+j⟩:Address)∈tile tag h s rb cb nr nc rows cols := by
  simp only [tile,List.mem_flatMap,List.mem_range,List.mem_filterMap]
  exact ⟨i,hi,j,hj,by simp [hr,hc]⟩

theorem program_member (nx ny nz : Nat) (body : Nat→Nat→Nat→List Address)
    (x y h : Nat) (a : Address) (hx : x<nx) (hy : y<ny) (hh : h<nz)
    (ha : a∈body x y h) : a∈programs nx ny nz body := by
  simp only [programs,List.mem_flatMap,List.mem_range]
  exact ⟨h,hh,y,hy,x,hx,ha⟩

theorem h_footprint (c : Shape) (flag : Bool) (a : Address) (ha : a∈eventsH c flag) : a.region∈Phase.h.writes := by
  simp only [eventsH,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,vb,hv,kb,hk,ha⟩
  rcases List.mem_append.mp ha with ha|ha
  · rcases List.mem_flatMap.mp ha with ⟨t,ht,ha⟩
    rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
    simp [Phase.writes]
  · cases flag <;> simp only [Bool.false_eq_true,↓reduceIte,List.not_mem_nil] at ha
    rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
    simp [Phase.writes]
theorem o_footprint (c : Shape) (a : Address) (ha : a∈eventsO c) : a.region∈Phase.o.writes := by
  simp only [eventsO,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,t,ht,vb,hv,ha⟩
  rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
  simp [Phase.writes]
theorem dh_footprint (c : Shape) (a : Address) (ha : a∈eventsDH c) : a.region∈Phase.dh.writes := by
  simp only [eventsDH,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,vb,hv,kb,hk,t,ht,ha⟩
  rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
  simp [Phase.writes]
theorem qkv_footprint (c : Shape) (a : Address) (ha : a∈eventsQKV c) : a.region∈Phase.qkv.writes := by
  simp only [eventsQKV,programs,List.mem_flatMap,List.mem_range,List.mem_append] at ha
  rcases ha with ⟨h,hh,t,ht,kb,hk,ha⟩
  rcases ha with (ha|ha)|ha
  · rcases ha with ⟨vb,hv,ha⟩
    rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
    simp [Phase.writes]
  · rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
    simp [Phase.writes]
  · rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
    simp [Phase.writes]
theorem sum_footprint (c : Shape) (a : Address) (ha : a∈eventsSum c) : a.region∈Phase.sum.writes := by
  simp only [eventsSum,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨z,hz,y,hy,h,hh,ha⟩
  rw [tile_tag _ _ _ _ _ _ _ _ _ a ha]
  simp [Phase.writes]

theorem history_cover (c : Shape) (flag : Bool) (h t e j : Nat)
    (hh : h<c.heads) (ht : t<c.chunks) (he : e<c.K) (hj : j<c.V) :
    (⟨.history,h,t,e,j⟩:Address)∈eventsH c flag := by
  apply program_member _ _ _ _ (e/c.BK) (j/c.BV) h _
    (Shape.feature_block _ _ _ c.key_tile_pos he) (Shape.feature_block _ _ _ c.value_tile_pos hj) hh
  apply List.mem_append_left
  apply List.mem_flatMap.mpr
  refine ⟨t,List.mem_range.mpr ht,?_⟩
  have h := tile_member .history h t (e/c.BK*c.BK) (j/c.BV*c.BV) c.BK c.BV c.K c.V
    (e%c.BK) (j%c.BV) (Nat.mod_lt e c.key_tile_pos) (Nat.mod_lt j c.value_tile_pos)
  simpa [Nat.div_add_mod'] using h (by simpa [Nat.div_add_mod'] using he) (by simpa [Nat.div_add_mod'] using hj)

theorem final_cover (c : Shape) (h e j : Nat) (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    (⟨.finalState,h,0,e,j⟩:Address)∈eventsH c Bool.true := by
  apply program_member _ _ _ _ (e/c.BK) (j/c.BV) h _
    (Shape.feature_block _ _ _ c.key_tile_pos he) (Shape.feature_block _ _ _ c.value_tile_pos hj) hh
  apply List.mem_append_right
  have hm := tile_member .finalState h 0 (e/c.BK*c.BK) (j/c.BV*c.BV) c.BK c.BV c.K c.V
    (e%c.BK) (j%c.BV) (Nat.mod_lt e c.key_tile_pos) (Nat.mod_lt j c.value_tile_pos)
  simpa [Nat.div_add_mod'] using hm (by simpa [Nat.div_add_mod'] using he) (by simpa [Nat.div_add_mod'] using hj)

theorem output_cover (c : Shape) (h r j : Nat) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    (⟨.output,h,0,r,j⟩:Address)∈eventsO c := by
  apply program_member _ _ _ _ (j/c.BV) (r/64) h _
    (Shape.feature_block _ _ _ c.value_tile_pos hj) (c.chunk_of_row r hr) hh
  have hm := tile_member .output h 0 (r/64*64) (j/c.BV*c.BV) 64 c.BV c.T c.V
    (r%64) (j%c.BV) (Nat.mod_lt r (by decide)) (Nat.mod_lt j c.value_tile_pos)
  simpa [Nat.div_add_mod'] using hm (by simpa [Nat.div_add_mod'] using hr) (by simpa [Nat.div_add_mod'] using hj)

theorem dhistory_cover (c : Shape) (h t e j : Nat)
    (hh : h<c.heads) (ht : t<c.chunks) (he : e<c.K) (hj : j<c.V) :
    (⟨.dhistory,h,t,e,j⟩:Address)∈eventsDH c := by
  apply program_member _ _ _ _ (e/c.BK) (j/c.BV) h _
    (Shape.feature_block _ _ _ c.key_tile_pos he) (Shape.feature_block _ _ _ c.value_tile_pos hj) hh
  apply List.mem_flatMap.mpr
  refine ⟨c.chunks-1-t,List.mem_range.mpr (by omega),?_⟩
  rw [show c.chunks-1-(c.chunks-1-t)=t by omega]
  have hm := tile_member .dhistory h t (e/c.BK*c.BK) (j/c.BV*c.BV) c.BK c.BV c.K c.V
    (e%c.BK) (j%c.BV) (Nat.mod_lt e c.key_tile_pos) (Nat.mod_lt j c.value_tile_pos)
  simpa [Nat.div_add_mod'] using hm (by simpa [Nat.div_add_mod'] using he) (by simpa [Nat.div_add_mod'] using hj)

theorem dq_cover (c : Shape) (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    (⟨.dq,h,0,r,e⟩:Address)∈eventsQKV c := by
  apply program_member _ _ _ _ (e/c.BK) (r/64) h _
    (Shape.feature_block _ _ _ c.key_tile_pos he) (c.chunk_of_row r hr) hh
  apply List.mem_append_left
  apply List.mem_append_right
  have hm := tile_member .dq h 0 (r/64*64) (e/c.BK*c.BK) 64 c.BK c.T c.K
    (r%64) (e%c.BK) (Nat.mod_lt r (by decide)) (Nat.mod_lt e c.key_tile_pos)
  simpa [Nat.div_add_mod'] using hm (by simpa [Nat.div_add_mod'] using hr) (by simpa [Nat.div_add_mod'] using he)

theorem dk_cover (c : Shape) (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    (⟨.dk,h,0,r,e⟩:Address)∈eventsQKV c := by
  apply program_member _ _ _ _ (e/c.BK) (r/64) h _
    (Shape.feature_block _ _ _ c.key_tile_pos he) (c.chunk_of_row r hr) hh
  apply List.mem_append_right
  have hm := tile_member .dk h 0 (r/64*64) (e/c.BK*c.BK) 64 c.BK c.T c.K
    (r%64) (e%c.BK) (Nat.mod_lt r (by decide)) (Nat.mod_lt e c.key_tile_pos)
  simpa [Nat.div_add_mod'] using hm (by simpa [Nat.div_add_mod'] using hr) (by simpa [Nat.div_add_mod'] using he)

theorem partial_cover (c : Shape) (b h r j : Nat)
    (hb : b<c.keyBlocks) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    (⟨.dvPart,h,b,r,j⟩:Address)∈eventsQKV c := by
  apply program_member _ _ _ _ b (r/64) h _ hb (c.chunk_of_row r hr) hh
  apply List.mem_append_left
  apply List.mem_append_left
  apply List.mem_flatMap.mpr
  refine ⟨j/c.BV,List.mem_range.mpr (Shape.feature_block _ _ _ c.value_tile_pos hj),?_⟩
  have hm := tile_member .dvPart h b (r/64*64) (j/c.BV*c.BV) 64 c.BV c.T c.V
    (r%64) (j%c.BV) (Nat.mod_lt r (by decide)) (Nat.mod_lt j c.value_tile_pos)
  simpa [Nat.div_add_mod'] using hm (by simpa [Nat.div_add_mod'] using hr) (by simpa [Nat.div_add_mod'] using hj)

theorem sum_cover (c : Shape) (h r j : Nat) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    (⟨.dv,h,0,r,j⟩:Address)∈eventsSum c := by
  apply program_member _ _ _ _ h 0 0 _ hh (by decide) (by decide)
  simpa using tile_member .dv h 0 0 0 c.T c.V c.T c.V r j hr hj (by simpa using hr) (by simpa using hj)
end Source

#axiomsClean Source.history_cover
#axiomsClean Source.dhistory_cover
#axiomsClean Source.partial_cover
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001


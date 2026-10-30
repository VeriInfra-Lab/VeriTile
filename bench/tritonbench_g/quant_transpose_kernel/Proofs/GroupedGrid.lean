import bench.tritonbench_g.quant_transpose_kernel.Proofs.GroupedMapping

namespace QuantTransposeFinite
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem active_index (p : Parameters) (pid lane : Nat) (h : active p pid lane) :
    offset p pid lane<p.n ∧ inputOffset p (offset p pid lane)=laneInput p pid lane := by
  have he := encode_decode p.M (row p.M p.N pid lane) (col p.M p.N pid lane) h.1
  exact ⟨active_encode_bound _ _ _ _ h.1 h.2,by simp [inputOffset,offset,laneInput,he]⟩

theorem footprint_complete (p : Parameters) (c : Config) (i : Nat) (hi : i<p.n) :
    i∈footprint p (List.range (programs p c)) := by
  obtain ⟨hr,hc,he⟩ := decode_bounds p.M p.N i hi
  obtain ⟨pid,hp,lane,hl,hm,hn⟩ := coordinate_preimage p.M p.N (i%p.M) (i/p.M) hr hc
  apply List.mem_flatMap.mpr
  refine ⟨pid,List.mem_range.mpr hp,?_⟩
  apply List.mem_filterMap.mpr
  refine ⟨lane,List.mem_range.mpr hl,?_⟩
  simp [active,offset,hm,hn,hr,hc,he]

/-- Every source lane coordinate, including masked lanes, lies within the
padded tile extent; masks are applied only to actual loads and stores. -/
theorem lane_bounds (p : Parameters) (c : Config) (pid lane : Nat)
    (hn : 0<p.N) (hp : pid<programs p c) (hl : lane<128*128) :
    row p.M p.N pid lane<gridM p.M*128 ∧ col p.M p.N pid lane<gridN p.N*128 := by
  have hg : 0<gridN p.N := by dsimp [gridN]; omega
  obtain ⟨_,hr,hc⟩ := tile_bounds (gridM p.M) (gridN p.N) pid hg hp
  dsimp [row,col]
  omega

/-- Distinct active source lanes in one whole grid have distinct output
addresses, including small M where inactive padded rows would collide. -/
theorem output_ownership (p : Parameters) (c : Config) (pid lane pid' lane' : Nat)
    (hn : 0<p.N) (hp : pid<programs p c) (hp' : pid'<programs p c)
    (hl : lane<128*128) (hl' : lane'<128*128)
    (ha : active p pid lane) (ha' : active p pid' lane')
    (he : offset p pid lane=offset p pid' lane') : pid=pid' ∧ lane=lane' := by
  have hd := encode_decode p.M (row p.M p.N pid lane) (col p.M p.N pid lane) ha.1
  have hd' := encode_decode p.M (row p.M p.N pid' lane') (col p.M p.N pid' lane') ha'.1
  have hr : row p.M p.N pid lane=row p.M p.N pid' lane' := by
    have hh := congrArg (fun i => i%p.M) he
    dsimp only [offset] at hh
    rw [hd.1,hd'.1] at hh
    exact hh
  have hc : col p.M p.N pid lane=col p.M p.N pid' lane' := by
    have hh := congrArg (fun i => i/p.M) he
    dsimp only [offset] at hh
    rw [hd.2,hd'.2] at hh
    exact hh
  have hrt : tileRow (gridM p.M) (gridN p.N) pid=tileRow (gridM p.M) (gridN p.N) pid' := by
    dsimp [row] at hr; omega
  have hct : tileCol (gridM p.M) (gridN p.N) pid=tileCol (gridM p.M) (gridN p.N) pid' := by
    dsimp [col] at hc; omega
  have hgn : 0<gridN p.N := by dsimp [gridN]; omega
  have hepid := tile_injective (gridM p.M) (gridN p.N) pid pid' hgn hp hp' hrt hct
  refine ⟨hepid,?_⟩
  subst pid'
  dsimp [row] at hr
  dsimp [col] at hc
  omega

#axiomsClean active_index
#axiomsClean footprint_complete
#axiomsClean lane_bounds
#axiomsClean output_ownership
#guard_msgs (drop info) in
#auditModuleAxioms
end QuantTransposeFinite

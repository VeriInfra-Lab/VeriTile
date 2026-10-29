import bench.repairs.tritonbench_g.bmm_optimized.Proofs.Bounds

namespace VeriTile.Bench.Repairs.BmmOptimized

theorem blocks_divisible (n tile : Nat) (ht : 0<tile) (hd : n%tile=0) :
    blocks n tile*tile=n := by
  obtain ⟨q,hq⟩ := Nat.dvd_of_mod_eq_zero hd
  subst n
  have rearrange : tile*q+tile-1=(tile-1)+q*tile := by rw [Nat.mul_comm q tile];omega
  unfold blocks
  rw [rearrange,Nat.add_mul_div_right _ _ ht,Nat.div_eq_of_lt (show tile-1<tile by omega)]
  simp [Nat.mul_comm]

theorem tiled_lane_bound (n tile t lane : Nat) (ht : 0<tile)
    (hb : t<blocks n tile) (hl : lane<tile) : t*tile+lane<blocks n tile*tile := by
  have mul := Nat.mul_le_mul_right tile (Nat.succ_le_of_lt hb)
  nlinarith

/-- Exact source heuristic: divisibility elides a mask only when all formed
lanes are in range. This covers every one of the eight M/N/K flag profiles. -/
def axisMask (n tile coordinate : Nat) : Prop := if n%tile=0 then True else coordinate<n
theorem axisMask_exact (n tile coordinate : Nat) (ht : 0<tile)
    (formed : coordinate<blocks n tile*tile) : axisMask n tile coordinate ↔ coordinate<n := by
  unfold axisMask
  by_cases h : n%tile=0
  · have active := formed
    rw [blocks_divisible n tile ht h] at active
    simp [h,active]
  · simp [h]

theorem source_lane_tiles (s : Shape) (c : Config) (pid : Program) (hp : pid∈programs s c)
    (lane : Lane) (hl : lane∈lanes c) :
    (cellOf s c pid lane).row<blocks s.m c.tm*c.tm ∧
    (cellOf s c pid lane).column<blocks s.n c.tn*c.tn ∧
    (cellOf s c pid lane).batch<s.batch := by
  have hpid := List.mem_product.mp hp
  have hrest := List.mem_product.mp hpid.2
  have hs := List.mem_product.mp hl
  have cb := config_bounds c
  have grid := VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.coordinates_in_grid
    pid.1 pid.2.1 (s.gx c) (s.gy c) c.group cb.2.2.2.2.1
    (List.mem_range.mp hpid.1) (List.mem_range.mp hrest.1)
  exact ⟨tiled_lane_bound s.m c.tm _ _ cb.1 grid.1 (List.mem_range.mp hs.1),
    tiled_lane_bound s.n c.tn _ _ cb.2.2.1 grid.2 (List.mem_range.mp hs.2),
    List.mem_range.mp hrest.2⟩

theorem source_masks (s : Shape) (c : Config) (pid : Program) (hp : pid∈programs s c)
    (lane : Lane) (hl : lane∈lanes c) (iteration k : Nat)
    (hi : iteration<s.iterations) (hk : k<32) :
    (axisMask s.m c.tm (cellOf s c pid lane).row ∧ axisMask s.k 32 (iteration*32+k) ↔
      (cellOf s c pid lane).row<s.m ∧ iteration*32+k<s.k) ∧
    (axisMask s.k 32 (iteration*32+k) ∧ axisMask s.n c.tn (cellOf s c pid lane).column ↔
      iteration*32+k<s.k ∧ (cellOf s c pid lane).column<s.n) ∧
    (axisMask s.m c.tm (cellOf s c pid lane).row ∧ axisMask s.n c.tn (cellOf s c pid lane).column ↔
      s.Active (cellOf s c pid lane)) := by
  have cb := config_bounds c
  have formed := source_lane_tiles s c pid hp lane hl
  have km := axisMask_exact s.k 32 _ (by decide) (tiled_lane_bound s.k 32 iteration k (by decide) hi hk)
  have rm := axisMask_exact s.m c.tm _ cb.1 formed.1
  have cm := axisMask_exact s.n c.tn _ cb.2.2.1 formed.2.1
  simp only [rm,km,cm,Shape.Active,formed.2.2,true_and,and_self]

theorem active_a_bound (s : Shape) (b row k : Nat) (hb : b<s.batch) (hr : row<s.m) (hk : k<s.k) :
    s.a b row k<s.batch*s.m*s.k :=
  address_bound ⟨s.batch,s.m,s.k,s.k⟩ ⟨b,row,k⟩ ⟨hb,hr,hk⟩
theorem active_b_bound (s : Shape) (b k col : Nat) (hb : b<s.batch) (hk : k<s.k) (hc : col<s.n) :
    s.b b k col<s.batch*s.k*s.n :=
  address_bound ⟨s.batch,s.k,s.n,s.k⟩ ⟨b,k,col⟩ ⟨hb,hk,hc⟩

/-- Every source pointer expression, including masked lanes and the final
post-loop pointer advance, is nonnegative and below signed32 overflow. -/
theorem formed_addresses (s : Shape) (c : Config) (abi : IndexABI s)
    (pid : Program) (hp : pid∈programs s c) (lane : Lane) (hl : lane∈lanes c)
    (iteration k : Nat) (hi : iteration≤s.iterations) (hk : k<32) :
    s.a pid.2.2 (cellOf s c pid lane).row (iteration*32+k)<2^31 ∧
    s.b pid.2.2 (iteration*32+k) (cellOf s c pid lane).column<2^31 ∧
    s.o pid.2.2 (cellOf s c pid lane).row (cellOf s c pid lane).column<2^31 := by
  have hs := List.mem_product.mp hl
  have formed := source_lane_tiles s c pid hp lane hl
  have row := source_row_bound s c pid hp lane.1 (List.mem_range.mp hs.1)
  have col := source_column_bound s c pid hp lane.2 (List.mem_range.mp hs.2)
  have kk := source_k_bound s iteration k hi hk
  exact ⟨padded_a_signed32 s abi _ _ _ formed.2.2 row kk,
    padded_b_signed32 s abi _ _ _ formed.2.2 kk col,
    padded_output_signed32 s abi _ _ _ formed.2.2 row col⟩

#axiomsClean blocks_divisible
#axiomsClean source_masks
#axiomsClean active_a_bound
#axiomsClean active_b_bound
#axiomsClean formed_addresses

theorem source_group_intermediates (s : Shape) (c : Config) (abi : IndexABI s)
    (pid : Program) (hp : pid∈programs s c) :
    let linear := pid.1+pid.2.1*s.gx c
    let count := s.gy c*c.group
    let group := linear/count
    0<count ∧ linear%count<count ∧ group<2^31 ∧ group*c.group+c.group<2^31 ∧
      0<VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.groupSize (s.gx c) group c.group := by
  dsimp only
  have pp := List.mem_product.mp hp
  have hx := List.mem_range.mp pp.1
  have hy := List.mem_range.mp (List.mem_product.mp pp.2).1
  have cb := config_bounds c
  have gypos : 0<s.gy c := by omega
  have positive := Nat.mul_pos gypos cb.2.2.2.2.1
  have linear := VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.linear_index_bound
    pid.1 pid.2.1 (s.gx c) (s.gy c) hx hy
  have base := VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.group_base_bound
    _ (s.gx c) (s.gy c) c.group linear
  have groupPos := VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.group_size_positive
    (s.gx c) _ c.group cb.2.2.2.2.1 base
  have signed := grid_linear_signed32 s c abi pid.1 pid.2.1 hx hy
  have quotient := Nat.div_le_self (pid.1+pid.2.1*s.gx c) (s.gy c*c.group)
  have gm := grid_max s c
  have prod := Nat.mul_le_mul gm.1 gm.2
  have gx : s.gx c≤s.gx c*s.gy c := by nlinarith
  have limit := abi.grid
  refine ⟨positive,Nat.mod_lt _ positive,lt_of_le_of_lt quotient signed.1,?_,groupPos⟩
  nlinarith [cb.2.2.2.2.2]
#axiomsClean source_group_intermediates
end VeriTile.Bench.Repairs.BmmOptimized

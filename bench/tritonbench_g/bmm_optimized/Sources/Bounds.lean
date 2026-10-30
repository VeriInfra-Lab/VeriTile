import bench.tritonbench_g.bmm_optimized.Sources.Grid

namespace VeriTile.Bench.Repairs.BmmOptimized

/-- Shape-only inequalities checked before allocation/launch. They include
the source's final post-loop pointer advance, not merely active load indices. -/
structure IndexABI (s : Shape) : Prop where
  batch : s.batch<2^31
  rows : s.m+128<2^31
  columns : s.n+128<2^31
  contraction : s.k+64<2^31
  grid : 2*blocks s.m 32*blocks s.n 32+2<2^31
  a : (s.batch*s.m+128)*s.k+64<2^31
  b : (s.batch*s.k+64)*s.n+128<2^31
  output : (s.batch*s.m+128)*s.n+128<2^31

theorem blocks_upper (n tile : Nat) (ht : 0<tile) : blocks n tile*tile<n+tile := by
  have h := Nat.div_mul_le_self (n+tile-1) tile
  unfold blocks
  omega
theorem source_row_bound (s : Shape) (c : Config) (pid : Program)
    (hp : pid∈programs s c) (lane : Nat) (hl : lane<c.tm) :
    rowTile s c pid*c.tm+lane<s.m+128 := by
  have h := List.mem_product.mp hp
  have hx := List.mem_range.mp h.1
  have hy := List.mem_range.mp (List.mem_product.mp h.2).1
  have cb := config_bounds c
  have tile := (VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.coordinates_in_grid
    pid.1 pid.2.1 (s.gx c) (s.gy c) c.group cb.2.2.2.2.1 hx hy).1
  have cover := Nat.mul_le_mul_right c.tm (Nat.succ_le_of_lt tile)
  have upper := blocks_upper s.m c.tm cb.1
  change rowTile s c pid<s.gx c at tile
  change (VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile
    pid.1 pid.2.1 (s.gx c) (s.gy c) c.group+1)*c.tm≤_ at cover
  change (rowTile s c pid+1)*c.tm≤blocks s.m c.tm*c.tm at cover
  nlinarith [cb.2.1]
theorem source_column_bound (s : Shape) (c : Config) (pid : Program)
    (hp : pid∈programs s c) (lane : Nat) (hl : lane<c.tn) :
    columnTile s c pid*c.tn+lane<s.n+128 := by
  have h := List.mem_product.mp hp
  have hx := List.mem_range.mp h.1
  have hy := List.mem_range.mp (List.mem_product.mp h.2).1
  have cb := config_bounds c
  have tile := (VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.coordinates_in_grid
    pid.1 pid.2.1 (s.gx c) (s.gy c) c.group cb.2.2.2.2.1 hx hy).2
  have cover := Nat.mul_le_mul_right c.tn (Nat.succ_le_of_lt tile)
  have upper := blocks_upper s.n c.tn cb.2.2.1
  change (columnTile s c pid+1)*c.tn≤blocks s.n c.tn*c.tn at cover
  nlinarith [cb.2.2.2.1]

theorem source_k_bound (s : Shape) (iteration lane : Nat)
    (hi : iteration≤s.iterations) (hl : lane<32) : iteration*32+lane<s.k+64 := by
  have h := Nat.mul_le_mul_right 32 hi
  have upper := padded_bound s.k
  change iteration*32≤((s.k+31)/32)*32 at h
  omega

theorem batch_row_bound (s : Shape) (b row : Nat) (hb : b<s.batch) (hr : row<s.m+128) :
    b*s.m+row+1≤s.batch*s.m+128 := by
  have h := Nat.mul_le_mul_right s.m (Nat.succ_le_of_lt hb)
  nlinarith
theorem padded_a_signed32 (s : Shape) (abi : IndexABI s) (b row k : Nat)
    (hb : b<s.batch) (hr : row<s.m+128) (hk : k<s.k+64) :
    s.a b row k<2^31 := by
  have h := Nat.mul_le_mul_right s.k (batch_row_bound s b row hb hr)
  have upper := abi.a
  unfold Shape.a
  nlinarith
theorem padded_b_signed32 (s : Shape) (abi : IndexABI s) (b k col : Nat)
    (hb : b<s.batch) (hk : k<s.k+64) (hc : col<s.n+128) :
    s.b b k col<2^31 := by
  have batch := Nat.mul_le_mul_right s.k (Nat.succ_le_of_lt hb)
  have base : b*s.k+k+1≤s.batch*s.k+64 := by nlinarith
  have h := Nat.mul_le_mul_right s.n base
  have upper := abi.b
  unfold Shape.b
  nlinarith
theorem padded_output_signed32 (s : Shape) (abi : IndexABI s) (b row col : Nat)
    (hb : b<s.batch) (hr : row<s.m+128) (hc : col<s.n+128) :
    s.o b row col<2^31 := by
  have h := Nat.mul_le_mul_right s.n (batch_row_bound s b row hb hr)
  have upper := abi.output
  unfold Shape.o
  nlinarith
theorem output_byte_signed64 (s : Shape) (abi : IndexABI s) (x : Cell)
    (hx : s.Active x) (i : Fin 4) : 4*s.o x.batch x.row x.column+i.val<2^63 := by
  unfold Shape.Active at hx
  have h := padded_output_signed32 s abi x.batch x.row x.column hx.1 (by omega) (by omega)
  omega

theorem config_min (c : Config) : 32≤c.tm ∧ 32≤c.tn := by
  cases c with | mk t st => cases t <;> simp [Config.tm,Config.tn]
theorem blocks_le_of_cover (n tile q : Nat) (ht : 0<tile) (cover : n≤q*tile) :
    blocks n tile≤q := by
  have h : (n+tile-1)/tile<q+1 := by
    apply (Nat.div_lt_iff_lt_mul ht).mpr
    have pred : n+tile-1+1=n+tile := by omega
    nlinarith
  exact Nat.le_of_lt_succ h
theorem grid_max (s : Shape) (c : Config) :
    s.gx c≤blocks s.m 32 ∧ s.gy c≤blocks s.n 32 := by
  have cb := config_bounds c
  have lo := config_min c
  constructor
  · apply blocks_le_of_cover s.m c.tm _ cb.1
    exact le_trans (VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMath.blocks_cover s.m 32 (by decide))
      (Nat.mul_le_mul_left _ lo.1)
  · apply blocks_le_of_cover s.n c.tn _ cb.2.2.1
    exact le_trans (VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMath.blocks_cover s.n 32 (by decide))
      (Nat.mul_le_mul_left _ lo.2)
theorem grid_linear_signed32 (s : Shape) (c : Config) (abi : IndexABI s) (px py : Nat)
    (hx : px<s.gx c) (hy : py<s.gy c) :
    px+py*s.gx c<2^31 ∧ s.gy c*c.group<2^31 := by
  have cb := config_bounds c
  have bounds := grid_max s c
  have prod := Nat.mul_le_mul bounds.1 bounds.2
  have linear := VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.linear_index_bound
    px py (s.gx c) (s.gy c) hx hy
  have gybound : s.gy c*c.group≤2*(s.gx c*s.gy c) := by
    have gxpos : 1≤s.gx c := by omega
    nlinarith [cb.2.2.2.2.2]
  have limit := abi.grid
  constructor <;> nlinarith

#axiomsClean source_row_bound
#axiomsClean source_column_bound
#axiomsClean source_k_bound
#axiomsClean padded_a_signed32
#axiomsClean padded_b_signed32
#axiomsClean padded_output_signed32
#axiomsClean output_byte_signed64
#axiomsClean grid_linear_signed32
end VeriTile.Bench.Repairs.BmmOptimized

import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Compiler

namespace VeriTile.Bench.Repairs.LayerNormTriton.Geometry
open VeriTile.Bench.Repairs.FastLayernorm.Storage

theorem one_forward_iteration (N B : Nat) (hn : 0<N) (hb : N≤B) :
    (N+B-1)/B=1 := by
  have h1 := (ceil_mem N B 0 (by omega)).mpr (by simpa using hn)
  have h2 : ¬1<(N+B-1)/B := by
    rw [ceil_mem N B 1 (by omega)]
    omega
  omega

/-- Even inactive column expressions fit signed int64. Only masked active
lanes access memory; absolute-pointer behavior is the ordinary load/store API. -/
theorem raw_row_offset_safe (x : Dense) (M N row col B : Nat)
    (shape : x.count=M*N) (hr : row<M) (hcol : col<B) (hb : B≤16384) : row*N+col<2^63 := by
  have hc := dense_count_bound x
  have hmul := Nat.mul_le_mul_right N (Nat.le_of_lt hr)
  norm_num at *
  omega

theorem group_tile_counter_safe (N M iteration lane : Nat)
    (hi : iteration<(min (groups N) M+31)/32) (hl : lane<32) : iteration*32+lane<288 := by
  have hg := groups_bounds N
  have hm := Nat.mod_lt (min (groups N) M+31) (by decide : 0<32)
  have hd := Nat.div_add_mod (min (groups N) M+31) 32
  omega

theorem final_column_safe (N pid lane : Nat) (hn : N≤16384)
    (hp : pid<(N+127)/128) (hl : lane<128) : pid*128+lane<16512 := by
  have hm := Nat.mod_lt (N+127) (by decide : 0<128)
  have hd := Nat.div_add_mod (N+127) 128
  omega

theorem final_raw_offset_safe (N row col : Nat) (hn : N≤16384) (hr : row<288) (hc : col<16512) :
    row*N+col<2^63 := by
  have hm := Nat.mul_le_mul hr.le hn
  norm_num at *
  omega

#axiomsClean raw_row_offset_safe
#axiomsClean final_raw_offset_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormTriton.Geometry

import bench.tritonbench_g.int8_quantization.Proofs.Geometry

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Native
open VeriTile.Triton
open Int8Quantization.Words (signed64 signed64_exact)
open Int8Quantization.Words (int32 int32_exact)
noncomputable section
set_option maxHeartbeats 6000000

/-- Native arithmetic facts derived by Direct.native_admitted from the
explicit block/tile/input/scale guards. No source index bound is deferred to
compiler success or a numerical assumption. -/
structure Admitted (p : Parameters) : Prop where
  blocks : p.blocks=Geometry.blocks p.length p.blockRows
  blockPositive : 0< p.blockRows
  channelsPositive : 0< p.channels
  covers : p.blockRows*p.channels≤ p.width
  compiledWidth : p.width≤1048576
  inputAllocation : 4*(p.batches*p.length*p.channels)≤2^63-1
  scaleAllocation : 4*(p.batches*p.blocks)≤2^63-1

theorem tile_start (length block tile : Nat) (hb : 0< block)
    (ht : tile< Geometry.blocks length block) : tile*block< length := by
  by_contra h
  have bound : length+block-1<(tile+1)*block := by
    have hsub : length+block-1< length+block := by omega
    nlinarith
  have quotient := (Nat.div_lt_iff_lt_mul hb).mpr bound
  unfold Geometry.blocks at ht
  omega

theorem bounds (p : Parameters) (a : Admitted p) (pid lane : Nat)
    (hg : pid+p.base< p.batches*p.blocks) (hl : lane< p.width) :
    let g:=pid+p.base
    let b:=g/p.blocks
    let t:=g%p.blocks
    let nativePrefix:=b*p.length*p.channels+t*p.blockRows*p.channels
    0< p.blocks ∧ b< p.batches ∧ t*p.blockRows< p.length ∧
      g<2^61 ∧ p.base<2^61 ∧ p.blocks<2^61 ∧
      p.length<2^61 ∧ p.channels≤1048576 ∧ p.blockRows≤1048576 ∧
      b*p.length≤ p.batches*p.length*p.channels ∧
      b*p.length*p.channels≤ p.batches*p.length*p.channels ∧
      t*p.blockRows*p.channels< p.batches*p.length*p.channels ∧
      nativePrefix< p.batches*p.length*p.channels ∧
      t*p.blockRows+lane/p.channels<2^63 ∧ nativePrefix+lane<2^63 := by
  dsimp
  have hb : 0< p.batches := by nlinarith
  have hblocks : 0< p.blocks := by nlinarith
  have batch : (pid+p.base)/p.blocks< p.batches := (Nat.div_lt_iff_lt_mul hblocks).mpr hg
  have tile : (pid+p.base)%p.blocks< Geometry.blocks p.length p.blockRows := by
    rw [←a.blocks];exact Nat.mod_lt _ hblocks
  have start:=tile_start p.length p.blockRows _ a.blockPositive tile
  have length : 0< p.length := by omega
  have count : p.batches*p.length*p.channels<2^61 := by have:=a.inputAllocation;omega
  have scaled : p.batches*p.blocks<2^61 := by have:=a.scaleAllocation;omega
  have clen : p.length≤ p.batches*p.length*p.channels :=
    le_trans (Nat.le_mul_of_pos_left _ hb) (Nat.le_mul_of_pos_right _ a.channelsPositive)
  have cblocks : p.blocks≤ p.batches*p.blocks := Nat.le_mul_of_pos_left _ hb
  have cb : p.blockRows≤ p.blockRows*p.channels := Nat.le_mul_of_pos_right _ a.channelsPositive
  have cc : p.channels≤ p.blockRows*p.channels := Nat.le_mul_of_pos_left _ a.blockPositive
  have batchProduct : ((pid+p.base)/p.blocks+1)*p.length*p.channels≤ p.batches*p.length*p.channels :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (by omega))
  have rowProduct : (pid+p.base)/p.blocks*p.length≤
      (pid+p.base)/p.blocks*p.length*p.channels := Nat.le_mul_of_pos_right _ a.channelsPositive
  have startProduct : (pid+p.base)%p.blocks*p.blockRows*p.channels< p.length*p.channels :=
    Nat.mul_lt_mul_of_pos_right start a.channelsPositive
  have plen : p.length*p.channels≤ p.batches*p.length*p.channels := by
    simpa [Nat.mul_assoc] using Nat.le_mul_of_pos_left (p.length*p.channels) hb
  have nativePrefix : (pid+p.base)/p.blocks*p.length*p.channels+
      (pid+p.base)%p.blocks*p.blockRows*p.channels< p.batches*p.length*p.channels := by
    nlinarith
  have divlane := Nat.div_le_self lane p.channels
  have width := a.compiledWidth
  have cover := a.covers
  refine ⟨hblocks,batch,start,lt_trans hg scaled,?_,lt_of_le_of_lt cblocks scaled,
    lt_of_le_of_lt clen count,by omega,by omega,?_,?_,lt_of_lt_of_le startProduct plen,nativePrefix,?_,?_⟩ <;> omega

structure Prefix where
  group : Int
  batch : Int
  tile : Int
  lane : Int
  row : Int
  offset : Int
  deriving DecidableEq

/-- Literal widened native address nativePrefix, including inactive lanes. -/
def nativePrefix (p : Parameters) (pid lane : Nat) : Prefix :=
  let g:=signed64 (signed64 (int32 pid)+(p.base:Int))
  let b:=g/(p.blocks:Int)
  let t:=g%(p.blocks:Int)
  let l:=signed64 (int32 lane)
  let row:=signed64 (signed64 (t*(p.blockRows:Int))+l/(p.channels:Int))
  let off:=signed64 (signed64 (signed64 (signed64 (b*(p.length:Int))*(p.channels:Int))+
    signed64 (signed64 (t*(p.blockRows:Int))*(p.channels:Int)))+l)
  ⟨g,b,t,l,row,off⟩

def exactPrefix (p : Parameters) (pid lane : Nat) : Prefix :=
  let g:=pid+p.base
  ⟨g,g/p.blocks,g%p.blocks,lane,g%p.blocks*p.blockRows+lane/p.channels,
    g/p.blocks*p.length*p.channels+g%p.blocks*p.blockRows*p.channels+lane⟩

theorem prefix_exact (p : Parameters) (a : Admitted p) (pid lane : Nat)
    (pidLimit : pid<2^31) (hg : pid+p.base< p.batches*p.blocks) (hl : lane< p.width) :
    nativePrefix p pid lane=exactPrefix p pid lane := by
  obtain ⟨blocks,batch,start,g,base,blocksLimit,length,channels,block,
    bprod,bprod2,tprod,pre,row,off⟩:=bounds p a pid lane hg hl
  have count : p.batches*p.length*p.channels<2^61 := by have:=a.inputAllocation;omega
  have laneLimit : lane<2^31 := by have:=a.compiledWidth;omega
  have s64 (n : Nat) (h : n<2^63) : signed64 (n:Int)=(n:Int) := signed64_exact _ (by omega) (by omega)
  unfold nativePrefix exactPrefix
  rw [int32_exact _ (by omega) (by omega),s64 pid (by omega),
      int32_exact _ (by omega) (by omega),s64 lane (by omega)]
  simp only [←Nat.cast_add,←Nat.cast_mul,←Int.natCast_ediv,←Int.natCast_emod]
  rw [s64 (pid+p.base) (by omega)]
  simp only [←Nat.cast_add,←Nat.cast_mul,←Int.natCast_ediv,←Int.natCast_emod]
  rw [s64 ((pid+p.base)%p.blocks*p.blockRows) (by omega)]
  simp only [←Nat.cast_add,←Nat.cast_mul,←Int.natCast_ediv,←Int.natCast_emod]
  rw [s64 ((pid+p.base)%p.blocks*p.blockRows+lane/p.channels) row,
    s64 ((pid+p.base)/p.blocks*p.length) (by omega)]
  simp only [←Nat.cast_add,←Nat.cast_mul]
  rw [s64 ((pid+p.base)/p.blocks*p.length*p.channels) (by omega),
    s64 ((pid+p.base)%p.blocks*p.blockRows*p.channels) (by omega)]
  simp only [←Nat.cast_add]
  rw [s64 ((pid+p.base)/p.blocks*p.length*p.channels+(pid+p.base)%p.blocks*p.blockRows*p.channels) (by omega)]
  simp only [←Nat.cast_add]
  rw [s64 _ off]

def pointer (bytes base : Nat) (offset : Int) : Nat :=
  (BitVec.ofInt 64 ((base:Int)+(bytes:Int)*offset)).toNat

theorem pointer_exact (bytes base count offset : Nat) (positive : 0< bytes)
    (capacity : base+bytes*count≤2^64) (active : offset< count) :
    pointer bytes base offset=base+bytes*offset := by
  have bound : base+bytes*offset<2^64 := by
    have:=Nat.mul_lt_mul_of_pos_left active positive;omega
  simp only [pointer,BitVec.toNat_ofInt]
  rw [Int.emod_eq_of_lt (by positivity) (by exact_mod_cast bound)]
  norm_cast

#axiomsClean bounds
#axiomsClean prefix_exact
#axiomsClean pointer_exact
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Native

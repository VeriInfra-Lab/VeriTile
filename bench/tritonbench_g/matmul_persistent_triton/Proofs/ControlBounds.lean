import bench.tritonbench_g.matmul_persistent_triton.Proofs.Bounds

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
set_option maxHeartbeats 5000000

/-- Bounds for the actual two separate loop trip counts. No product of the
tile and K counts is evaluated by the repaired program. -/
theorem loop_bounds (p : Problem) (b : Buffers) (layout : Layout p b)
    (hs : 0<p.SMS) (pid : Nat) :
    p.rows<2^63 ∧ p.columns<2^63 ∧ p.kTiles<2^63 ∧
    p.work pid≤p.tiles ∧ p.work pid<2^63 := by
  have mb := layout.aBound.rows
  have nb := layout.bBound.columns
  have kb := layout.aBound.columns
  rw [layout.aShape.1] at mb
  rw [layout.bShape.2] at nb
  rw [layout.aShape.2] at kb
  have r := ceil_le_length p.M 128 (by omega)
  have c := ceil_le_length p.N 256 (by omega)
  have widths : p.profile.blockK=64 ∨ p.profile.blockK=128 ∨ p.profile.blockK=256 := by
    cases p.profile <;> simp [Storage.blockK]
  have k := ceil_le_length p.K p.profile.blockK widths
  have t := (p.tile_count_bound b layout).1
  have div := Nat.div_le_self p.tiles p.SMS
  have mod := Nat.mod_add_div p.tiles p.SMS
  rw [Nat.mul_comm p.SMS] at mod
  have scale := Nat.mul_le_mul_left (p.tiles/p.SMS) (show 1≤p.SMS by omega)
  simp only [Nat.mul_one] at scale
  have work : p.work pid≤p.tiles := by
    unfold Problem.work Schedule.tilesFor
    split_ifs <;> omega
  exact ⟨lt_of_le_of_lt r mb,lt_of_le_of_lt c nb,lt_of_le_of_lt k kb,work,lt_of_le_of_lt work t⟩

/-- Every multiplication/addition/division operand in the persistent/group
control path lies in the nonnegative signed64 range. The subtracted group
base is also bounded by rows, so the source subtraction is nonnegative. -/
theorem group_intermediates (p : Problem) (b : Buffers) (layout : Layout p b)
    (hm : 0<p.M) (hs : 0<p.SMS) (pid ordinal : Nat)
    (hp : pid<p.grid) (hq : ordinal<p.work pid) :
    let tile := p.tile pid ordinal
    let span := p.group*p.columns
    let base := tile/span*p.group
    let size := min (p.rows-base) p.group
    ordinal*p.SMS<2^63 ∧ tile<2^63 ∧ 0<span ∧ span<2^63 ∧
    tile/span<2^63 ∧ base<p.rows ∧ base<2^63 ∧
    p.rows-base<2^63 ∧ 0<size ∧ size<2^63 ∧
    tile%size<2^63 ∧ base+tile%size<2^63 ∧
    tile%span<2^63 ∧ tile%span/size<2^63 := by
  dsimp only
  have pidBound : pid<p.SMS := lt_of_lt_of_le hp (Nat.min_le_left _ _)
  have tileBound : p.tile pid ordinal<p.tiles :=
    (Schedule.ordinal_bound_iff p.tiles p.SMS pid ordinal hs pidBound).mp hq
  have tc := p.tile_count_bound b layout
  have lc := loop_bounds p b layout hs pid
  have gp := p.group_positive hm
  have cp : 0<p.columns := by
    change p.tile pid ordinal<p.rows*p.columns at tileBound
    by_contra hz
    have z : p.columns=0 := by omega
    simp [z] at tileBound
  have sp := Nat.mul_pos gp cp
  have gc := Schedule.grouped_bounds p.rows p.columns p.group (p.tile pid ordinal) gp tileBound
  have mul := Nat.div_mul_le_self (p.tile pid ordinal) (p.group*p.columns)
  have base : p.tile pid ordinal/(p.group*p.columns)*p.group<p.rows := by
    change p.tile pid ordinal<p.rows*p.columns at tileBound
    nlinarith
  have rem := Nat.mod_lt (p.tile pid ordinal) gc.1
  have remspan := Nat.mod_lt (p.tile pid ordinal) sp
  have div := Nat.div_le_self (p.tile pid ordinal) (p.group*p.columns)
  have size := Nat.min_le_left (p.rows-p.tile pid ordinal/(p.group*p.columns)*p.group) p.group
  change 0<min (p.rows-p.tile pid ordinal/(p.group*p.columns)*p.group) p.group ∧
    p.tile pid ordinal/(p.group*p.columns)*p.group+p.tile pid ordinal%
      min (p.rows-p.tile pid ordinal/(p.group*p.columns)*p.group) p.group<p.rows ∧
    p.tile pid ordinal%(p.group*p.columns)/
      min (p.rows-p.tile pid ordinal/(p.group*p.columns)*p.group) p.group<p.columns at gc
  change p.tile pid ordinal%min (p.rows-p.tile pid ordinal/(p.group*p.columns)*p.group) p.group<
    min (p.rows-p.tile pid ordinal/(p.group*p.columns)*p.group) p.group at rem
  have mulBound : ordinal*p.SMS<2^63 := by
    change pid+ordinal*p.SMS<p.tiles at tileBound
    omega
  exact ⟨mulBound,tileBound.trans tc.1,sp,tc.2,lt_of_le_of_lt div (tileBound.trans tc.1),
    base,base.trans lc.1,by omega,gc.1,by omega,by omega,
    gc.2.1.trans lc.1,remspan.trans tc.2,gc.2.2.trans lc.2.1⟩

/-- Signed K-mask subtraction is representable and has exactly the original
tail predicate. In particular inactive raw K lanes never select a load. -/
theorem k_control (p : Problem) (b : Buffers) (layout : Layout p b)
    (iteration lane : Nat) (hi : iteration<p.kTiles) (hl : lane<p.profile.blockK) :
    iteration*p.profile.blockK<2^63 ∧
    -(2^63 : Int)≤(p.K : Int)-(iteration : Int)*p.profile.blockK ∧
    (p.K : Int)-(iteration : Int)*p.profile.blockK<2^63 ∧
    ((lane : Int)<(p.K : Int)-(iteration : Int)*p.profile.blockK ↔ p.k iteration lane<p.K) := by
  have kb := layout.aBound.columns
  rw [layout.aShape.2] at kb
  have widths : p.profile.blockK=64 ∨ p.profile.blockK=128 ∨ p.profile.blockK=256 := by
    cases p.profile <;> simp [Storage.blockK]
  have raw := padded_bound p.K p.profile.blockK iteration lane kb widths hi hl
  have product : iteration*p.profile.blockK<2^63 := by omega
  have cast : ((iteration*p.profile.blockK : Nat) : Int)=(iteration : Int)*p.profile.blockK := by simp
  refine ⟨product,?_,?_,RealMath.signed_tail_mask _ _ _ _⟩ <;> omega

/-- Byte scaling is checked at the actual tensor data pointer, after its
storage offset. Thus signed64 element arithmetic and final unsigned64 pointer
formation both follow from ordinary Tensor backing bounds. -/
theorem Tensor.pointer_products (t : Tensor) (bounded : t.Bounded) (row column : Nat)
    (hr : row<t.rows) (hc : column<t.columns) :
    row*t.rowStride<2^63 ∧ column*t.columnStride<2^63 ∧
    row*t.rowStride+column*t.columnStride<2^63 ∧
    t.dtype.bytes*(row*t.rowStride+column*t.columnStride)<2^63 ∧
    t.pointer.toNat+t.dtype.bytes*(row*t.rowStride+column*t.columnStride)<2^64 := by
  have formed := t.formed_products bounded row column hr hc
  have logical := t.logical_bound bounded row column hr hc
  have lower : row*t.rowStride+column*t.columnStride≤(t.address row column).2 := by
    simp only [Tensor.address]; omega
  have scaled := Nat.mul_le_mul_left t.dtype.bytes lower
  have cap := bounded.capacity
  have exactPointer : t.pointer.toNat+t.dtype.bytes*(row*t.rowStride+column*t.columnStride)=
      t.allocationBase.toNat+t.dtype.bytes*(t.address row column).2 := by
    have pointer := bounded.pointer
    simp only [show ¬(t.rows=0 ∨ t.columns=0) by omega,if_false] at pointer
    rw [pointer]
    simp only [Tensor.address,Nat.mul_add]
    omega
  exact ⟨formed.1,formed.2.1,formed.2.2.1,by omega,by rw [exactPointer]; exact formed.2.2.2⟩

#axiomsClean group_intermediates
#axiomsClean k_control
#axiomsClean Tensor.pointer_products
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulPersistentTriton

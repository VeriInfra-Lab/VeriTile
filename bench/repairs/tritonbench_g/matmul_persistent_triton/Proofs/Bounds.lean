import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Types
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Geometry

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 5000000

theorem bytes_positive (storage : Storage) : 0<storage.bytes := by cases storage <;> decide
theorem blockK_positive (storage : Storage) : 0<storage.blockK := by cases storage <;> decide

theorem source_ceil (length width : Nat) (widths : width=64 ∨ width=128 ∨ width=256) :
    length/width+(if length%width=0 then 0 else 1)=RealMath.blocks length width := by
  rcases widths with rfl|rfl|rfl <;> unfold RealMath.blocks <;> split_ifs <;> omega

theorem ceil_le_length (length width : Nat) (widths : width=64 ∨ width=128 ∨ width=256) :
    RealMath.blocks length width≤length := by
  rcases widths with rfl|rfl|rfl <;> unfold RealMath.blocks <;> omega

theorem padded_bound (length width tile lane : Nat) (lengthBound : length<2^63)
    (widths : width=64 ∨ width=128 ∨ width=256)
    (tileBound : tile<RealMath.blocks length width) (laneBound : lane<width) :
    tile*width+lane<2^63 := by
  rcases widths with rfl|rfl|rfl <;> unfold RealMath.blocks at tileBound <;> norm_num at * <;> omega

theorem clamp_bound (length coordinate : Nat) (positive : 0<length) : clamp length coordinate<length := by
  unfold clamp
  split_ifs <;> omega

theorem Tensor.logical_bound (t : Tensor) (bounded : t.Bounded) (r c : Nat)
    (hr : r<t.rows) (hc : c<t.columns) :
    t.dtype.bytes*(t.address r c).2+t.dtype.bytes≤t.allocationBytes ∧
      (t.address r c).2<2^63 := by
  have hs := bounded.span
  have rr := Nat.mul_le_mul_right t.rowStride (show r≤t.rows-1 by omega)
  have cc := Nat.mul_le_mul_right t.columnStride (show c≤t.columns-1 by omega)
  have positive := bytes_positive t.dtype
  simp only [Tensor.span,show ¬(t.rows=0 ∨ t.columns=0) by omega,if_false] at hs
  have le : (t.address r c).2+1≤t.offset+(t.rows-1)*t.rowStride+(t.columns-1)*t.columnStride+1 := by
    simp only [Tensor.address]
    omega
  have scale := Nat.mul_le_mul_left t.dtype.bytes le
  have cap := bounded.capacity
  constructor <;> nlinarith

theorem Tensor.formed_products (t : Tensor) (bounded : t.Bounded) (r c : Nat)
    (hr : r<t.rows) (hc : c<t.columns) :
    r*t.rowStride<2^63 ∧ c*t.columnStride<2^63 ∧
    r*t.rowStride+c*t.columnStride<2^63 ∧
    t.allocationBase.toNat+t.dtype.bytes*(t.address r c).2<2^64 := by
  have h := t.logical_bound bounded r c hr hc
  have positive := bytes_positive t.dtype
  have physical := bounded.physical
  simp only [Tensor.address] at h ⊢
  constructor
  · omega
  constructor
  · omega
  constructor
  · omega
  · omega

theorem Layout.output_count (p : Problem) (b : Buffers) (layout : Layout p b) : p.M*p.N<2^63 := by
  have hs := layout.cBound.span
  have hc := layout.cBound.capacity
  have hp := bytes_positive b.C.dtype
  rw [Tensor.span,layout.cShape.1,layout.cShape.2] at hs
  by_cases empty : p.M=0 ∨ p.N=0
  · rcases empty with h|h <;> simp [h]
  · simp only [empty,if_false,layout.outputStrides.1,layout.outputStrides.2,Nat.mul_one] at hs
    rw [max_eq_left (show 1≤p.N by omega)] at hs
    have m : p.M-1+1=p.M := by omega
    have n : p.N-1+1=p.N := by omega
    have product := congrArg (fun x => x*p.N) m
    simp only [Nat.add_mul,Nat.one_mul] at product
    have unscale := Nat.mul_le_mul_right
      (b.C.offset+(p.M-1)*p.N+(p.N-1)+1) (show 1≤b.C.dtype.bytes by omega)
    simp only [Nat.one_mul] at unscale
    have spanBound := unscale.trans hs
    omega

theorem Problem.group_positive (p : Problem) (hm : 0<p.M) : 0<p.group := by
  exact lt_min (by decide) (RealMath.blocks_positive p.M 128 hm (by decide))

theorem Problem.tile_count_bound (p : Problem) (b : Buffers) (layout : Layout p b) :
    p.tiles<2^63 ∧ p.group*p.columns<2^63 := by
  have rows := ceil_le_length p.M 128 (by omega)
  have columns := ceil_le_length p.N 256 (by omega)
  have total := layout.output_count p b
  have mul := Nat.mul_le_mul rows columns
  have groups := Nat.mul_le_mul_right p.columns (Nat.min_le_right 8 p.rows)
  change p.rows*p.columns<2^63 ∧ _
  change p.rows≤p.M at rows
  change p.columns≤p.N at columns
  change p.rows*p.columns≤p.M*p.N at mul
  change p.group*p.columns≤p.rows*p.columns at groups
  constructor <;> omega

/-- Concrete source-formed coordinates and clamped pointer arithmetic are
bounded before any generic grid-value theorem is used. Even inactive K/output
lanes form a valid ordinary backing address, because clamping precedes it. -/
theorem formed_addresses (p : Problem) (b : Buffers) (layout : Layout p b)
    (hm : 0<p.M) (hn : 0<p.N) (hk : 0<p.K) (hs : 0<p.SMS)
    (pid ordinal iteration rm cn kl : Nat)
    (hp : pid<p.grid) (hq : ordinal<p.work pid) (hi : iteration<p.kTiles)
    (hr : rm<128) (hc : cn<256) (hl : kl<p.profile.blockK) :
    let tile := p.tile pid ordinal
    let row := p.row tile rm
    let column := p.column tile cn
    let kval := p.k iteration kl
    tile<p.tiles ∧ row<2^63 ∧ column<2^63 ∧ kval<2^63 ∧
    (b.A.address (clamp p.M row) (clamp p.K kval)).2<2^63 ∧
    (b.B.address (clamp p.K kval) (clamp p.N column)).2<2^63 ∧
    (b.C.address (clamp p.M row) (clamp p.N column)).2<2^63 := by
  dsimp only
  have pidBound : pid<p.SMS := lt_of_lt_of_le hp (Nat.min_le_left _ _)
  have tileBound : p.tile pid ordinal<p.tiles :=
    (Schedule.ordinal_bound_iff p.tiles p.SMS pid ordinal hs pidBound).mp hq
  have group := Schedule.grouped_bounds p.rows p.columns p.group (p.tile pid ordinal)
    (p.group_positive hm) tileBound
  have mBound : p.M<2^63 := by rw [← layout.aShape.1]; exact layout.aBound.rows
  have nBound : p.N<2^63 := by rw [← layout.bShape.2]; exact layout.bBound.columns
  have kBound : p.K<2^63 := by rw [← layout.aShape.2]; exact layout.aBound.columns
  have widths : p.profile.blockK=64 ∨ p.profile.blockK=128 ∨ p.profile.blockK=256 := by cases p.profile <;> simp [Storage.blockK]
  have ra := clamp_bound p.M (p.row (p.tile pid ordinal) rm) hm
  have cb := clamp_bound p.N (p.column (p.tile pid ordinal) cn) hn
  have kk := clamp_bound p.K (p.k iteration kl) hk
  refine ⟨tileBound,padded_bound p.M 128 _ rm mBound (by omega) group.2.1 hr,
    padded_bound p.N 256 _ cn nBound (by omega) group.2.2 hc,
    padded_bound p.K p.profile.blockK iteration kl kBound widths hi hl,?_,?_,?_⟩
  · exact (b.A.logical_bound layout.aBound _ _ (layout.aShape.1 ▸ ra) (layout.aShape.2 ▸ kk)).2
  · exact (b.B.logical_bound layout.bBound _ _ (layout.bShape.1 ▸ kk) (layout.bShape.2 ▸ cb)).2
  · exact (b.C.logical_bound layout.cBound _ _ (layout.cShape.1 ▸ ra) (layout.cShape.2 ▸ cb)).2

#axiomsClean formed_addresses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton

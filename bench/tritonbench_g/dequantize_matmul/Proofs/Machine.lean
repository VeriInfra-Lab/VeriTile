import bench.tritonbench_g.dequantize_matmul.Proofs.Selection
import bench.tritonbench_g.dequantize_matmul.Proofs.JIT

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 3000000

def gridCount (extent block : Nat) := (extent+block-1)/block
def padded (extent block : Nat) := gridCount extent block*block
def Lane.launched (s : Shape) (l : Lane tile) : Prop :=
  l.programK<gridCount s.K tile.rows ∧ l.programN<gridCount s.N tile.columns

/-- Ordinary launch ABI, including every formed tail address. Contiguity does
not imply these inequalities: in particular singleton inert strides may be
arbitrarily large. No value or completed output occurs in this contract.
The Python launcher may specialize scalar arguments as i32 or i64; in either
case its local argument decoder must preserve these nonnegative integers. -/
structure IndexABI (s : Shape) (tile : Tile) (db df : Dense2 s.K s.N) (ds : Dense1 s.N) : Prop where
  dimensions : s.K<2^63 ∧ s.N<2^63
  strides : db.rowStride<2^63 ∧ db.colStride<2^63 ∧
    df.rowStride<2^63 ∧ df.colStride<2^63 ∧ ds.stride<2^63
  gridX : gridCount s.K tile.rows≤2^31-1
  gridY : gridCount s.N tile.columns≤65535
  coordinates : padded s.K tile.rows<2^63 ∧ padded s.N tile.columns<2^63
  bFormed : padded s.K tile.rows*db.rowStride+padded s.N tile.columns*db.colStride<2^63
  fpbBytesFormed : 4*(padded s.K tile.rows*df.rowStride+padded s.N tile.columns*df.colStride)<2^63
  scaleBytesFormed : 4*(padded s.N tile.columns*ds.stride)<2^63

theorem lane_padded (l : Lane tile) (h : l.launched s) :
    l.k<padded s.K tile.rows ∧ l.n<padded s.N tile.columns := by
  constructor
  · have hm := Nat.mul_le_mul_right tile.rows h.1
    have hl := l.localK.isLt
    simp only [Nat.succ_mul] at hm
    change l.programK*tile.rows+l.localK.val<gridCount s.K tile.rows*tile.rows
    omega
  · have hm := Nat.mul_le_mul_right tile.columns h.2
    have hl := l.localN.isLt
    simp only [Nat.succ_mul] at hm
    change l.programN*tile.columns+l.localN.val<gridCount s.N tile.columns*tile.columns
    omega

theorem signed64_exact (x : Nat) (h : x<2^63) : (BitVec.ofNat 64 x).toInt=(x : Int) := by
  have hx : x<2^64 := by omega
  have hn : (BitVec.ofNat 64 x).toNat=x := by rw [BitVec.toNat_ofNat]; exact Nat.mod_eq_of_lt hx
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

def coordinate64 (program block laneIndex : Nat) : BitVec 64 :=
  BitVec.ofNat 64 program*BitVec.ofNat 64 block+BitVec.ofNat 64 laneIndex
def affine64 (k n rowStride colStride : Nat) : BitVec 64 :=
  BitVec.ofNat 64 k*BitVec.ofNat 64 rowStride+BitVec.ofNat 64 n*BitVec.ofNat 64 colStride

theorem coordinate64_exact (program block laneIndex : Nat) (h : program*block+laneIndex<2^63) :
    (coordinate64 program block laneIndex).toInt=(program*block+laneIndex : Nat) := by
  simpa only [coordinate64,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat] using signed64_exact _ h
theorem affine64_exact (k n rowStride colStride : Nat) (h : k*rowStride+n*colStride<2^63) :
    (affine64 k n rowStride colStride).toInt=(k*rowStride+n*colStride : Nat) := by
  simpa only [affine64,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat] using signed64_exact _ h

/-- These are equalities for the actual coordinate and affine forms, including
masked tail lanes. Their nonnegative intermediate products are bounded by
the same sums, so interpreting the typed AST's Int operations does not erase
an overflow. Pointer base addition/ownership remains the local memory ABI. -/
def IndexPost (s : Shape) (tile : Tile) (db df : Dense2 s.K s.N) (ds : Dense1 s.N) : Prop :=
  ∀ l : Lane tile,l.launched s →
    (coordinate64 l.programK tile.rows l.localK.val).toInt=(l.k : Int) ∧
    (coordinate64 l.programN tile.columns l.localN.val).toInt=(l.n : Int) ∧
    (affine64 l.k l.n db.rowStride db.colStride).toInt=(l.k*db.rowStride+l.n*db.colStride : Nat) ∧
    (affine64 l.k l.n df.rowStride df.colStride).toInt=(l.k*df.rowStride+l.n*df.colStride : Nat) ∧
    (BitVec.ofNat 64 l.n*BitVec.ofNat 64 ds.stride).toInt=(l.n*ds.stride : Nat) ∧
    4*(l.k*df.rowStride+l.n*df.colStride)<2^63 ∧ 4*(l.n*ds.stride)<2^63

theorem IndexABI.exact (abi : IndexABI s tile db df ds) : IndexPost s tile db df ds := by
  intro l launch
  obtain ⟨hk,hn⟩ := lane_padded l launch
  have hb := Nat.add_le_add (Nat.mul_le_mul_right db.rowStride (Nat.le_of_lt hk))
    (Nat.mul_le_mul_right db.colStride (Nat.le_of_lt hn))
  have hf := Nat.add_le_add (Nat.mul_le_mul_right df.rowStride (Nat.le_of_lt hk))
    (Nat.mul_le_mul_right df.colStride (Nat.le_of_lt hn))
  have hs := Nat.mul_le_mul_right ds.stride (Nat.le_of_lt hn)
  have hf4 := Nat.mul_le_mul_left 4 hf
  have hs4 := Nat.mul_le_mul_left 4 hs
  have bf := abi.fpbBytesFormed
  have bs := abi.scaleBytesFormed
  refine ⟨coordinate64_exact _ _ _ (hk.trans abi.coordinates.1),
    coordinate64_exact _ _ _ (hn.trans abi.coordinates.2),
    affine64_exact _ _ _ _ (hb.trans_lt abi.bFormed),
    affine64_exact _ _ _ _ (by omega),?_,by omega,by omega⟩
  rw [BitVec.ofNat_mul_ofNat]
  exact signed64_exact _ (by omega)

/-- Exact save/enter/restore control used by torch.cuda.device. The runtime's
local enter/exit operations implement this state update; the caller need not
already have the input's device current. -/
structure DeviceState (α : Type) where
  currentDevice : Nat
  payload : α
def deviceCall (owner : Nat) (launch : DeviceState α → DeviceState α) (caller : DeviceState α) :=
  let finished := launch ⟨owner,caller.payload⟩
  { finished with currentDevice := caller.currentDevice }
theorem deviceCall_restores (owner : Nat) (launch : DeviceState α → DeviceState α)
    (caller : DeviceState α) : (deviceCall owner launch caller).currentDevice=caller.currentDevice := rfl
theorem deviceCall_enters (owner : Nat) (launch : DeviceState α → DeviceState α)
    (caller : DeviceState α) : (deviceCall owner launch caller).payload=(launch ⟨owner,caller.payload⟩).payload := rfl

#axiomsClean lane_padded
#axiomsClean signed64_exact
#axiomsClean IndexABI.exact
#axiomsClean deviceCall_restores
#axiomsClean deviceCall_enters
end
end VeriTile.Bench.Repairs.DequantizeMatmul

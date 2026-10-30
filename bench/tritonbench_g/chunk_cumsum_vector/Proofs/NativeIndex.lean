import bench.tritonbench_g.chunk_cumsum_vector.Proofs.Geometry

/-! Word-level index safety uses actual storage windows. The program-id ABI
is separate from deferred launch-grid compatibility; it is not a shape filter.
Both inactive coordinates are clamped before either pointer is formed. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex
open VeriTile.Triton

def signedLimit : Nat := 9223372036854775808

structure Buffer where
  base : Nat
  extent : Nat
  bytes : Nat
  bytesPositive : 0<bytes
  allocation : extent*bytes<signedLimit
  placement : base+extent*bytes≤18446744073709551616

structure Backing (h : Call) where
  input : Buffer
  output : Buffer
  outputCells : h.B*h.H*h.T*h.S≤output.extent
  inputWindow : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
    h.inputStrides.address h.H row time channel < input.extent
  outputWindow : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
    h.outputStrides.address h.H row time channel < output.extent

structure PID (h : Call) extends Geometry.Program h where
  featureSigned32 : feature<2147483648
  rowSigned32 : row<2147483648

theorem extent_bound (b : Buffer) : b.extent<signedLimit :=
  lt_of_le_of_lt (Nat.le_mul_of_pos_right _ b.bytesPositive) b.allocation

theorem dimensions (h : Call) (c : Config) (hc : 0<c.BT) (backing : Backing h)
    (p : PID h) (q : Nat) (hq : q<Geometry.chunks h c) :
    h.H<signedLimit ∧ h.T<signedLimit ∧ h.S<signedLimit := by
  have timePos := Nat.zero_lt_of_lt (Geometry.chunk_start h c hc q hq)
  have featurePos := Nat.zero_lt_of_lt (Geometry.feature_start h p.toProgram)
  have rowPos := Nat.zero_lt_of_lt p.rowInside
  have batchPos : 0<h.B := by
    by_contra hn
    have hb : h.B=0 := by omega
    simp [hb] at rowPos
  have extentTotal := lt_of_le_of_lt backing.outputCells (extent_bound backing.output)
  have rows : h.B*h.H≤h.B*h.H*h.T*h.S :=
    (Nat.le_mul_of_pos_right _ timePos).trans (Nat.le_mul_of_pos_right _ featurePos)
  have t : h.T≤h.B*h.H*h.T*h.S :=
    (Nat.le_mul_of_pos_left _ rowPos).trans (Nat.le_mul_of_pos_right _ featurePos)
  have s : h.S≤h.B*h.H*h.T*h.S := Nat.le_mul_of_pos_left _ (Nat.mul_pos rowPos timePos)
  exact ⟨((Nat.le_mul_of_pos_left _ batchPos).trans rows).trans_lt extentTotal,
    t.trans_lt extentTotal,s.trans_lt extentTotal⟩

theorem signed64 (n : Nat) (hn : n<signedLimit) : (BitVec.ofNat 64 n).toInt=(n:Int) := by
  have hnat : (BitVec.ofNat 64 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    dsimp only [signedLimit] at hn
    omega
  rw [BitVec.toInt_eq_toNat_of_lt,hnat]
  rw [hnat]
  dsimp only [signedLimit] at hn
  norm_num
  omega

theorem widen32 (n : Nat) (hn : n<2147483648) :
    BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt=BitVec.ofNat 64 n := by
  have hnat : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    omega
  have hint : (BitVec.ofNat 32 n).toInt=(n:Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt,hnat]
    rw [hnat]
    norm_num
    omega
  rw [hint,BitVec.ofInt_natCast]

theorem signed_divmod (a b : Nat) (ha : a<signedLimit) (hb : b<signedLimit) :
    (BitVec.ofNat 64 a).sdiv (BitVec.ofNat 64 b)=BitVec.ofNat 64 (a/b) ∧
    (BitVec.ofNat 64 a).srem (BitVec.ofNat 64 b)=BitVec.ofNat 64 (a%b) := by
  have ne : BitVec.ofNat 64 b≠-1#64 := by
    intro eq
    have hint := congrArg BitVec.toInt eq
    rw [signed64 b hb] at hint
    have neg : (-1#64).toInt=(-1:Int) := by decide
    rw [neg] at hint
    omega
  constructor
  · apply BitVec.eq_of_toInt_eq
    rw [BitVec.toInt_sdiv_of_ne_or_ne _ _ (Or.inr ne),signed64 a ha,signed64 b hb,
      signed64 (a/b) ((Nat.div_le_self _ _).trans_lt ha)]
    simp
  · apply BitVec.eq_of_toInt_eq
    rw [BitVec.toInt_srem,signed64 a ha,signed64 b hb,
      signed64 (a%b) ((Nat.mod_le _ _).trans_lt ha)]
    simp [Int.tmod_eq_emod]

def offsetWord (h : Call) (p : PID h) (strides : Strides) (time channel : Nat) : BitVec 64 :=
  let row := BitVec.ofInt 64 (BitVec.ofNat 32 p.row).toInt
  row.sdiv (BitVec.ofNat 64 h.H)*BitVec.ofNat 64 strides.batch+
    row.srem (BitVec.ofNat 64 h.H)*BitVec.ofNat 64 strides.head+
    BitVec.ofNat 64 time*BitVec.ofNat 64 strides.time+
    BitVec.ofNat 64 channel*BitVec.ofNat 64 strides.channel

theorem offsetWord_exact (h : Call) (p : PID h) (strides : Strides)
    (time channel : Nat) (head : h.H<signedLimit) :
    offsetWord h p strides time channel=BitVec.ofNat 64 (strides.address h.H p.row time channel) := by
  have row : p.row<signedLimit := by
    have := p.rowSigned32
    dsimp only [signedLimit]
    omega
  have qr := signed_divmod p.row h.H row head
  simp only [offsetWord,widen32 _ p.rowSigned32,qr.1,qr.2,←BitVec.ofNat_mul,←BitVec.ofNat_add,
    Strides.address,Strides.base]

theorem byte_span (buffer : Buffer) (offset : Nat) (inside : offset<buffer.extent) :
    buffer.base+offset*buffer.bytes+buffer.bytes≤18446744073709551616 := by
  have hp := Nat.mul_le_mul_right buffer.bytes (Nat.succ_le_of_lt inside)
  have := buffer.placement
  nlinarith

theorem pointer_word (buffer : Buffer) (offset : Nat) (inside : offset<buffer.extent) :
    (BitVec.ofNat 64 buffer.base+BitVec.ofNat 64 offset*BitVec.ofNat 64 buffer.bytes).toNat=
      buffer.base+offset*buffer.bytes := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  have span := byte_span buffer offset inside
  have positive := buffer.bytesPositive
  norm_num
  omega

theorem source_addresses (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (p : PID h) (q : Nat) (hq : q<Geometry.chunks h c)
    (i : Fin c.BT) (j : Fin BS) :
    let time := Geometry.time h c q i.val
    let channel := Geometry.column h p.feature j.val
    (offsetWord h p h.inputStrides time channel).toInt=(h.inputStrides.address h.H p.row time channel:Int) ∧
    (offsetWord h p h.outputStrides time channel).toInt=(h.outputStrides.address h.H p.row time channel:Int) ∧
    (BitVec.ofNat 64 backing.input.base+offsetWord h p h.inputStrides time channel*BitVec.ofNat 64 backing.input.bytes).toNat=
      backing.input.base+h.inputStrides.address h.H p.row time channel*backing.input.bytes ∧
    (BitVec.ofNat 64 backing.output.base+offsetWord h p h.outputStrides time channel*BitVec.ofNat 64 backing.output.bytes).toNat=
      backing.output.base+h.outputStrides.address h.H p.row time channel*backing.output.bytes := by
  have positive := (Kernels.configurations_positive c hc).1
  have dims := dimensions h c positive backing p q hq
  have time := Geometry.time_bound h c positive q hq i.val
  have channel := Geometry.column_bound h p.toProgram j.val
  have input := backing.inputWindow _ _ _ p.rowInside time channel
  have output := backing.outputWindow _ _ _ p.rowInside time channel
  dsimp only
  rw [offsetWord_exact h p h.inputStrides _ _ dims.1,offsetWord_exact h p h.outputStrides _ _ dims.1]
  exact ⟨signed64 _ (input.trans (extent_bound backing.input)),signed64 _ (output.trans (extent_bound backing.output)),
    pointer_word backing.input _ input,pointer_word backing.output _ output⟩

theorem source_indices (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (p : PID h) (q : Nat) (hq : q<Geometry.chunks h c)
    (i : Fin c.BT) (j : Fin BS) :
    Geometry.chunks h c<signedLimit ∧
    (BitVec.ofNat 64 q*BitVec.ofNat 64 c.BT).toInt=(q*c.BT:Int) ∧
    (BitVec.ofInt 64 (BitVec.ofNat 32 p.feature).toInt*BitVec.ofNat 64 BS).toInt=(p.feature*BS:Int) ∧
    (BitVec.ofNat 64 h.T-BitVec.ofNat 64 (q*c.BT)).toInt=(h.T-q*c.BT:Nat) ∧
    (BitVec.ofNat 64 h.S-BitVec.ofNat 64 (p.feature*BS)).toInt=(h.S-p.feature*BS:Nat) ∧
    ((BitVec.ofNat 64 i.val).slt (BitVec.ofNat 64 (h.T-q*c.BT))=decide (Geometry.activeTime h c q i.val)) ∧
    ((BitVec.ofNat 64 j.val).slt (BitVec.ofNat 64 (h.S-p.feature*BS))=decide (Geometry.activeColumn h p.feature j.val)) := by
  have positive := (Kernels.configurations_positive c hc).1
  have dims := dimensions h c positive backing p q hq
  have start := Geometry.chunk_start h c positive q hq
  have feature := Geometry.feature_start h p.toProgram
  have lane : i.val<signedLimit := by
    have := i.isLt
    have := (Kernels.configurations_positive c hc).2
    dsimp only [signedLimit]
    omega
  have column : j.val<signedLimit := by
    have := j.isLt
    dsimp only [BS,signedLimit] at *
    omega
  refine ⟨(Geometry.ceiling_le h.T c.BT positive).trans_lt dims.2.1,?_,?_,?_,?_,?_,?_⟩
  · rw [←BitVec.ofNat_mul]
    exact signed64 _ (start.trans dims.2.1)
  · rw [widen32 _ p.featureSigned32,←BitVec.ofNat_mul]
    exact signed64 _ (feature.trans dims.2.2)
  · rw [BitVec.ofNat_sub_ofNat_of_le h.T (q*c.BT) (by dsimp only [signedLimit] at dims; omega) (Nat.le_of_lt start)]
    exact signed64 _ ((Nat.sub_le _ _).trans_lt dims.2.1)
  · rw [BitVec.ofNat_sub_ofNat_of_le h.S (p.feature*BS) (by dsimp only [signedLimit] at dims; omega) (Nat.le_of_lt feature)]
    exact signed64 _ ((Nat.sub_le _ _).trans_lt dims.2.2)
  · rw [BitVec.slt_eq_decide,signed64 _ lane,signed64 _ ((Nat.sub_le _ _).trans_lt dims.2.1)]
    simp [Geometry.activeTime]
  · rw [BitVec.slt_eq_decide,signed64 _ column,signed64 _ ((Nat.sub_le _ _).trans_lt dims.2.2)]
    simp [Geometry.activeColumn]

/-- This is the source's select-before-add expression, including masked lanes. -/
def coordinateWord (length start lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 start +
    if (BitVec.ofNat 64 lane).slt (BitVec.ofNat 64 length-BitVec.ofNat 64 start)
    then BitVec.ofNat 64 lane else BitVec.ofNat 64 0

theorem coordinateWord_exact (length start lane : Nat)
    (inside : start<length) (bound : length<signedLimit) (laneBound : lane<signedLimit) :
    coordinateWord length start lane=BitVec.ofNat 64 (start+if lane<length-start then lane else 0) := by
  have sub : BitVec.ofNat 64 length-BitVec.ofNat 64 start=BitVec.ofNat 64 (length-start) :=
    BitVec.ofNat_sub_ofNat_of_le length start (by dsimp only [signedLimit] at bound; omega) (Nat.le_of_lt inside)
  unfold coordinateWord
  rw [sub,BitVec.slt_eq_decide,signed64 _ laneBound,
    signed64 _ ((Nat.sub_le _ _).trans_lt bound)]
  by_cases h : lane<length-start
  · have hi : (lane:Int)<(length-start:Nat) := by exact_mod_cast h
    simp [h,hi,BitVec.ofNat_add]
  · have hi : ¬(lane:Int)<(length-start:Nat) := by exact_mod_cast h
    simp [h,hi]

theorem source_coordinates (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (p : PID h) (q : Nat) (hq : q<Geometry.chunks h c)
    (i : Fin c.BT) (j : Fin BS) :
    coordinateWord h.T (q*c.BT) i.val=BitVec.ofNat 64 (Geometry.time h c q i.val) ∧
    coordinateWord h.S (p.feature*BS) j.val=BitVec.ofNat 64 (Geometry.column h p.feature j.val) := by
  have positive := (Kernels.configurations_positive c hc).1
  have dims := dimensions h c positive backing p q hq
  have lane : i.val<signedLimit := by
    have := i.isLt
    have := (Kernels.configurations_positive c hc).2
    dsimp only [signedLimit]
    omega
  have column : j.val<signedLimit := by
    have := j.isLt
    dsimp only [BS,signedLimit] at *
    omega
  exact ⟨coordinateWord_exact _ _ _ (Geometry.chunk_start h c positive q hq) dims.2.1 lane,
    coordinateWord_exact _ _ _ (Geometry.feature_start h p.toProgram) dims.2.2 column⟩

#axiomsClean source_coordinates
#axiomsClean source_addresses
#axiomsClean source_indices
end VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex

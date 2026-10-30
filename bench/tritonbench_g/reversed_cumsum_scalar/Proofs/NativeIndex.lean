import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Geometry

/-! Kernel integer safety follows from the actual tensor windows. The checked
wrapper supplies signed-i32 program identifiers; all subsequent arithmetic is
i64. This module does not address shared launcher grid-product compatibility. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.NativeIndex
open VeriTile.Triton
noncomputable section

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
  outputCells : h.B*h.H*h.T≤output.extent
  inputWindow : ∀ row t, row<h.B*h.H → t<h.T →
    h.inputStrides.address h.H row t< input.extent
  outputWindow : ∀ row t, row<h.B*h.H → t<h.T →
    h.outputStrides.address h.H row t< output.extent

/-- This is the native program_id ABI, not an input-shape restriction. -/
structure PID (h : Call) where
  value : Nat
  inside : value<h.B*h.H
  signed32 : value<2147483648

theorem extent_bound (buffer : Buffer) : buffer.extent<signedLimit :=
  lt_of_le_of_lt (Nat.le_mul_of_pos_right _ buffer.bytesPositive) buffer.allocation

theorem time_bound (h : Call) (backing : Backing h) (pid : PID h) : h.T<signedLimit := by
  have hp : 0<h.B*h.H := Nat.zero_lt_of_lt pid.inside
  exact lt_of_le_of_lt ((Nat.le_mul_of_pos_left h.T hp).trans backing.outputCells)
    (extent_bound backing.output)

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

theorem pid_cast (h : Call) (pid : PID h) :
    BitVec.ofInt 64 (BitVec.ofNat 32 pid.value).toInt=BitVec.ofNat 64 pid.value := by
  have hnat : (BitVec.ofNat 32 pid.value).toNat=pid.value := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    have := pid.signed32
    omega
  have hint : (BitVec.ofNat 32 pid.value).toInt=(pid.value:Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt,hnat]
    rw [hnat]
    norm_num
    have := pid.signed32
    omega
  rw [hint,BitVec.ofInt_natCast]

theorem address_parts (strides : Strides) (H row t bound : Nat)
    (hb : strides.address H row t<bound) :
    row/H*strides.batch<bound ∧ row%H*strides.head<bound ∧
    strides.base H row<bound ∧ t*strides.time<bound := by
  dsimp only [Strides.address,Strides.base] at *
  omega

/-- Every executed chunk's i64 counter/product, remaining-length subtraction,
lane predicate, clamped time and both pointer addresses are exact. This also
covers masked lanes: their address uses the valid chunk start. -/
theorem source_indices (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (pid : PID h) (q : Nat) (hq : q<Geometry.chunks h c)
    (i : Fin c.BT) :
    Geometry.chunks h c<signedLimit ∧ q*c.BT<h.T ∧
    (BitVec.ofNat 64 q*BitVec.ofNat 64 c.BT).toInt=(q*c.BT:Int) ∧
    (BitVec.ofNat 64 h.T-BitVec.ofNat 64 (q*c.BT)).toInt=(h.T-q*c.BT:Nat) ∧
    ((BitVec.ofNat 64 i.val).slt (BitVec.ofNat 64 (h.T-q*c.BT))=
      decide (Geometry.active h c q i)) ∧
    Geometry.time h c q i<h.T ∧
    (BitVec.ofNat 64 (Geometry.time h c q i)).toInt=(Geometry.time h c q i:Int) ∧
    h.inputStrides.address h.H pid.value (Geometry.time h c q i)<backing.input.extent ∧
    h.outputStrides.address h.H pid.value (Geometry.time h c q i)<backing.output.extent := by
  have positive := (Kernels.configurations_positive c hc).1
  have tBound := time_bound h backing pid
  have start := Geometry.chunk_start h c positive q hq
  have time := Geometry.time_bound h c positive q hq i
  have chunks := lt_of_le_of_lt (Geometry.chunks_le_time h c positive) tBound
  have lane : i.val<signedLimit := by
    have := i.isLt
    have := (Kernels.configurations_positive c hc).2
    dsimp only [signedLimit]
    omega
  refine ⟨chunks,start,?_,?_,?_,time,signed64 _ (time.trans tBound),
    backing.inputWindow _ _ pid.inside time,backing.outputWindow _ _ pid.inside time⟩
  · rw [←BitVec.ofNat_mul]
    exact signed64 _ (start.trans tBound)
  · rw [BitVec.ofNat_sub_ofNat_of_le h.T (q*c.BT) (by dsimp only [signedLimit] at tBound; omega) (Nat.le_of_lt start)]
    exact signed64 _ (lt_of_le_of_lt (Nat.sub_le _ _) tBound)
  · rw [BitVec.slt_eq_decide,signed64 _ lane,signed64 _ (lt_of_le_of_lt (Nat.sub_le _ _) tBound)]
    simp [Geometry.active]

/-- Full byte spans fit the ordinary allocation, including the last element. -/
theorem byte_span (buffer : Buffer) (offset : Nat) (inside : offset<buffer.extent) :
    buffer.base+offset*buffer.bytes+buffer.bytes≤18446744073709551616 := by
  have hp := Nat.mul_le_mul_right buffer.bytes (Nat.succ_le_of_lt inside)
  have := buffer.placement
  nlinarith

/-- Actual signed division/remainder on a nonnegative native coordinate. -/
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

def offsetWord (h : Call) (pid : PID h) (strides : Strides) (time : Nat) : BitVec 64 :=
  let row := BitVec.ofInt 64 (BitVec.ofNat 32 pid.value).toInt
  row.sdiv (BitVec.ofNat 64 h.H)*BitVec.ofNat 64 strides.batch+
    row.srem (BitVec.ofNat 64 h.H)*BitVec.ofNat 64 strides.head+
    BitVec.ofNat 64 time*BitVec.ofNat 64 strides.time

theorem head_bound (h : Call) (backing : Backing h) (pid : PID h) (ht : 0<h.T) :
    h.H<signedLimit := by
  have bh : 0<h.B := by
    by_contra hn
    have hp := pid.inside
    rw [Nat.eq_zero_of_not_pos hn,Nat.zero_mul] at hp
    omega
  have rows : h.B*h.H<signedLimit := lt_of_le_of_lt
    ((Nat.le_mul_of_pos_right _ ht).trans backing.outputCells) (extent_bound backing.output)
  exact lt_of_le_of_lt (Nat.le_mul_of_pos_left _ bh) rows

theorem offsetWord_exact (h : Call) (backing : Backing h) (pid : PID h)
    (strides : Strides) (time : Nat) (ht : time<h.T) :
    offsetWord h pid strides time=BitVec.ofNat 64 (strides.address h.H pid.value time) := by
  have row : pid.value<signedLimit := by
    have := pid.signed32
    dsimp only [signedLimit]
    omega
  have quot := signed_divmod pid.value h.H row (head_bound h backing pid (Nat.zero_lt_of_lt ht))
  simp only [offsetWord,pid_cast,quot.1,quot.2,←BitVec.ofNat_mul,←BitVec.ofNat_add,
    Strides.address,Strides.base]

theorem pointer_word (buffer : Buffer) (offset : Nat) (inside : offset<buffer.extent) :
    (BitVec.ofNat 64 buffer.base+BitVec.ofNat 64 offset*BitVec.ofNat 64 buffer.bytes).toNat=
      buffer.base+offset*buffer.bytes := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  have span := byte_span buffer offset inside
  have positive := buffer.bytesPositive
  norm_num
  omega

/-- Concrete word operations through signed div/mod, both stride products,
base additions, byte scaling and pointer addition project to the AST address. -/
theorem source_addresses (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (pid : PID h) (q : Nat) (hq : q<Geometry.chunks h c) (i : Fin c.BT) :
    let t := Geometry.time h c q i
    (offsetWord h pid h.inputStrides t).toInt=(h.inputStrides.address h.H pid.value t:Int) ∧
    (offsetWord h pid h.outputStrides t).toInt=(h.outputStrides.address h.H pid.value t:Int) ∧
    (BitVec.ofNat 64 backing.input.base+offsetWord h pid h.inputStrides t*BitVec.ofNat 64 backing.input.bytes).toNat=
      backing.input.base+h.inputStrides.address h.H pid.value t*backing.input.bytes ∧
    (BitVec.ofNat 64 backing.output.base+offsetWord h pid h.outputStrides t*BitVec.ofNat 64 backing.output.bytes).toNat=
      backing.output.base+h.outputStrides.address h.H pid.value t*backing.output.bytes := by
  have time := Geometry.time_bound h c (Kernels.configurations_positive c hc).1 q hq i
  have input := backing.inputWindow _ _ pid.inside time
  have output := backing.outputWindow _ _ pid.inside time
  dsimp only
  rw [offsetWord_exact h backing pid h.inputStrides _ time,offsetWord_exact h backing pid h.outputStrides _ time]
  exact ⟨signed64 _ (input.trans (extent_bound backing.input)),signed64 _ (output.trans (extent_bound backing.output)),
    pointer_word backing.input _ input,pointer_word backing.output _ output⟩

theorem clamped_time_word (h : Call) (c : Config) (q : Nat) (i : Fin c.BT) :
    BitVec.ofNat 64 (q*c.BT)+(if Geometry.active h c q i then BitVec.ofNat 64 i.val else 0)=
      BitVec.ofNat 64 (Geometry.time h c q i) := by
  by_cases active : Geometry.active h c q i <;>
    simp [Geometry.time,active,BitVec.ofNat_add]

/-- The signed reverse loop reaches -1 exactly, including the empty loop. -/
theorem decrement (n : Nat) (hn : n<signedLimit) :
    (BitVec.ofNat 64 n-1#64).toInt=(n:Int)-1 := by
  cases n with
  | zero => decide
  | succ n =>
    have one : (1#64)=BitVec.ofNat 64 1 := rfl
    rw [one,BitVec.ofNat_sub_ofNat_of_le (n+1) 1 (by decide) (by omega)]
    simp only [Nat.add_sub_cancel]
    rw [signed64 n (by omega)]
    simp

/-- Concrete native certificate for every actual program_id and configuration.
The launcher's global grid segmentation is separate from these kernel facts. -/
structure ProgramCertificate (h : Call) (c : Config) (b : Backing h) (pid : PID h) : Prop where
  pidWiden : BitVec.ofInt 64 (BitVec.ofNat 32 pid.value).toInt=BitVec.ofNat 64 pid.value
  countCast : (BitVec.ofNat 64 (Geometry.chunks h c)).toInt=(Geometry.chunks h c:Int)
  initialCounter : (BitVec.ofNat 64 (Geometry.chunks h c)-1#64).toInt=(Geometry.chunks h c:Int)-1
  decrementCounter : ∀ q, q<Geometry.chunks h c → (BitVec.ofNat 64 q-1#64).toInt=(q:Int)-1
  loopGuard : ∀ q, q<Geometry.chunks h c → (-1#64).slt (BitVec.ofNat 64 q)=Bool.true
  terminalGuard : (-1#64).slt (-1#64)=Bool.false
  start : ∀ q, q<Geometry.chunks h c → (BitVec.ofNat 64 q*BitVec.ofNat 64 c.BT).toInt=(q*c.BT:Int)
  remaining : ∀ q, q<Geometry.chunks h c →
    (BitVec.ofNat 64 h.T-BitVec.ofNat 64 (q*c.BT)).toInt=(h.T-q*c.BT:Nat)
  mask : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    (BitVec.ofNat 64 i.val).slt (BitVec.ofNat 64 (h.T-q*c.BT))=decide (Geometry.active h c q i)
  clampedTime : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    (BitVec.ofNat 64 (q*c.BT)+(if Geometry.active h c q i then BitVec.ofNat 64 i.val else 0)).toInt=
      (Geometry.time h c q i:Int)
  inputOffset : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    (offsetWord h pid h.inputStrides (Geometry.time h c q i)).toInt=
      (h.inputStrides.address h.H pid.value (Geometry.time h c q i):Int)
  outputOffset : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    (offsetWord h pid h.outputStrides (Geometry.time h c q i)).toInt=
      (h.outputStrides.address h.H pid.value (Geometry.time h c q i):Int)
  inputPointer : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    (BitVec.ofNat 64 b.input.base+offsetWord h pid h.inputStrides (Geometry.time h c q i)*BitVec.ofNat 64 b.input.bytes).toNat=
      b.input.base+h.inputStrides.address h.H pid.value (Geometry.time h c q i)*b.input.bytes
  outputPointer : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    (BitVec.ofNat 64 b.output.base+offsetWord h pid h.outputStrides (Geometry.time h c q i)*BitVec.ofNat 64 b.output.bytes).toNat=
      b.output.base+h.outputStrides.address h.H pid.value (Geometry.time h c q i)*b.output.bytes
  inputWindow : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    h.inputStrides.address h.H pid.value (Geometry.time h c q i)<b.input.extent
  outputWindow : ∀ q, q<Geometry.chunks h c → ∀ i : Fin c.BT,
    h.outputStrides.address h.H pid.value (Geometry.time h c q i)<b.output.extent

theorem certify (h : Call) (c : Config) (hc : c∈configurations) (b : Backing h) (pid : PID h) :
    ProgramCertificate h c b pid := by
  have count := lt_of_le_of_lt (Geometry.chunks_le_time h c (Kernels.configurations_positive c hc).1) (time_bound h b pid)
  refine ⟨pid_cast h pid,signed64 _ count,decrement _ count,
    fun q hq => decrement q (hq.trans count),?_,by decide,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro q hq
    rw [BitVec.slt_eq_decide,signed64 q (hq.trans count)]
    have neg : (-1#64).toInt=(-1:Int) := by decide
    rw [neg]
    simp only [decide_eq_true_eq]
    omega
  · intro q hq
    exact (source_indices h c hc b pid q hq ⟨0,(Kernels.configurations_positive c hc).1⟩).2.2.1
  · intro q hq
    exact (source_indices h c hc b pid q hq ⟨0,(Kernels.configurations_positive c hc).1⟩).2.2.2.1
  · intro q hq i
    exact (source_indices h c hc b pid q hq i).2.2.2.2.1
  · intro q hq i
    rw [clamped_time_word]
    exact (source_indices h c hc b pid q hq i).2.2.2.2.2.2.1
  · intro q hq i
    exact (source_addresses h c hc b pid q hq i).1
  · intro q hq i
    exact (source_addresses h c hc b pid q hq i).2.1
  · intro q hq i
    exact (source_addresses h c hc b pid q hq i).2.2.1
  · intro q hq i
    exact (source_addresses h c hc b pid q hq i).2.2.2
  · intro q hq i
    exact (source_indices h c hc b pid q hq i).2.2.2.2.2.2.2.1
  · intro q hq i
    exact (source_indices h c hc b pid q hq i).2.2.2.2.2.2.2.2

#print axioms certify
#print axioms source_addresses
#print axioms clamped_time_word

#print axioms source_indices
#print axioms byte_span
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.NativeIndex

import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Geometry
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.ReversedCumsum.NativeIndex
open VeriTile.Triton Geometry
open ReversedCumsumScalar.NativeIndex (signedLimit signed64 decrement pointer_word)
noncomputable section
set_option maxHeartbeats 3000000

abbrev Backing (h : Call) := ReversedCumsumScalar.NativeIndex.Backing h.flat

structure PID (h : Call) where
  row : Row h
  feature : FeatureBlock h
  rowABI : row.val<2147483648
  featureABI : feature.val<2147483648

def PID.rowPID (h : Call) (pid : PID h) : ReversedCumsumScalar.NativeIndex.PID h.flat :=
  ⟨pid.row.val,pid.row.isLt,pid.rowABI⟩

theorem row_widen (h : Call) (pid : PID h) :
    BitVec.ofInt 64 (BitVec.ofNat 32 pid.row.val).toInt=BitVec.ofNat 64 pid.row.val :=
  ReversedCumsumScalar.NativeIndex.pid_cast h.flat (pid.rowPID h)

theorem feature_widen (h : Call) (pid : PID h) :
    BitVec.ofInt 64 (BitVec.ofNat 32 pid.feature.val).toInt=BitVec.ofNat 64 pid.feature.val := by
  let view := {h.flat with B:=1,H:=featureChunks h}
  exact ReversedCumsumScalar.NativeIndex.pid_cast view ⟨pid.feature.val,by simpa [view] using pid.feature.isLt,pid.featureABI⟩

theorem product_bound (h : Call) (b : Backing h) (pid : PID h) : h.T*h.S<signedLimit :=
  ReversedCumsumScalar.NativeIndex.time_bound h.flat b (pid.rowPID h)

theorem time_bound (h : Call) (b : Backing h) (pid : PID h) : h.T<signedLimit := by
  have pos : 0<h.S := Nat.zero_lt_of_lt (Geometry.feature_start h pid.feature)
  exact (Nat.le_mul_of_pos_right h.T pos).trans_lt (product_bound h b pid)

theorem space_bound (h : Call) (c : Config) (positive : 0<c.BT) (b : Backing h) (pid : PID h)
    (q : Nat) (hq : q<chunks h c) : h.S<signedLimit := by
  have pos : 0<h.T := Nat.zero_lt_of_lt (Geometry.chunk_start h c positive q hq)
  exact (Nat.le_mul_of_pos_left h.S pos).trans_lt (product_bound h b pid)

def offsetWord (h : Call) (pid : PID h) (t s : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 pid.row.val).toInt*BitVec.ofNat 64 (h.T*h.S)+
    BitVec.ofNat 64 t*BitVec.ofNat 64 h.S+BitVec.ofNat 64 s

theorem offsetWord_exact (h : Call) (pid : PID h) (t s : Nat) :
    offsetWord h pid t s=BitVec.ofNat 64 (address h pid.row.val t s) := by
  simp only [offsetWord,row_widen,←BitVec.ofNat_mul,←BitVec.ofNat_add,address]

theorem windows (h : Call) (b : Backing h) (pid : PID h) (t s : Nat) (ht : t<h.T) (hs : s<h.S) :
    address h pid.row.val t s<b.input.extent ∧ address h pid.row.val t s<b.output.extent := by
  have hi := b.inputWindow pid.row.val (t*h.S+s) pid.row.isLt (inner_bound h t s ht hs)
  have ho := b.outputWindow pid.row.val (t*h.S+s) pid.row.isLt (inner_bound h t s ht hs)
  rw [(flat_address h _ _).1] at hi
  rw [(flat_address h _ _).2] at ho
  simpa [address,Nat.add_assoc] using And.intro hi ho

theorem time_word (h : Call) (c : Config) (q : Nat) (i : Fin c.BT) :
    BitVec.ofNat 64 (q*c.BT)+(if activeT h c q i then BitVec.ofNat 64 i.val else 0)=
      BitVec.ofNat 64 (Geometry.time h c q i) := by
  by_cases active : activeT h c q i <;> simp [Geometry.time,active,BitVec.ofNat_add]

theorem feature_word (h : Call) (f : Nat) (j : Fin BS) :
    BitVec.ofNat 64 (f*BS)+(if activeS h f j then BitVec.ofNat 64 j.val else 0)=
      BitVec.ofNat 64 (Geometry.feature h f j) := by
  by_cases active : activeS h f j <;> simp [Geometry.feature,active,BitVec.ofNat_add]

structure ProgramCertificate (h : Call) (c : Config) (b : Backing h) (pid : PID h) : Prop where
  rowCast : BitVec.ofInt 64 (BitVec.ofNat 32 pid.row.val).toInt=BitVec.ofNat 64 pid.row.val
  featureCast : BitVec.ofInt 64 (BitVec.ofNat 32 pid.feature.val).toInt=BitVec.ofNat 64 pid.feature.val
  countCast : (BitVec.ofNat 64 (chunks h c)).toInt=(chunks h c:Int)
  initial : (BitVec.ofNat 64 (chunks h c)-1#64).toInt=(chunks h c:Int)-1
  step : ∀ q, q<chunks h c → (BitVec.ofNat 64 q-1#64).toInt=(q:Int)-1
  loopGuard : ∀ q, q<chunks h c → (-1#64).slt (BitVec.ofNat 64 q)=Bool.true
  terminalGuard : (-1#64).slt (-1#64)=Bool.false
  nonnegative : ∀ q, q<chunks h c → (0#64).sle (BitVec.ofNat 64 q)=Bool.true
  clampedTime : ∀ q, q<chunks h c → ∀ i : Fin c.BT,
    (BitVec.ofNat 64 (q*c.BT)+(if activeT h c q i then BitVec.ofNat 64 i.val else 0)).toInt=
      (Geometry.time h c q i:Int)
  clampedFeature : ∀ q, q<chunks h c → ∀ j : Fin BS,
    (BitVec.ofNat 64 (pid.feature.val*BS)+(if activeS h pid.feature.val j then BitVec.ofNat 64 j.val else 0)).toInt=
      (Geometry.feature h pid.feature.val j:Int)
  body : ∀ q, q<chunks h c → ∀ i : Fin c.BT, ∀ j : Fin BS,
    let t := Geometry.time h c q i
    let s := Geometry.feature h pid.feature.val j
    (BitVec.ofNat 64 q*BitVec.ofNat 64 c.BT).toInt=(q*c.BT:Int) ∧
    (BitVec.ofNat 64 pid.feature.val*BitVec.ofNat 64 BS).toInt=(pid.feature.val*BS:Int) ∧
    (BitVec.ofNat 64 h.T-BitVec.ofNat 64 (q*c.BT)).toInt=(h.T-q*c.BT:Nat) ∧
    (BitVec.ofNat 64 h.S-BitVec.ofNat 64 (pid.feature.val*BS)).toInt=(h.S-pid.feature.val*BS:Nat) ∧
    (BitVec.ofNat 64 i.val).slt (BitVec.ofNat 64 (h.T-q*c.BT))=decide (activeT h c q i) ∧
    (BitVec.ofNat 64 j.val).slt (BitVec.ofNat 64 (h.S-pid.feature.val*BS))=decide (activeS h pid.feature.val j) ∧
    t<h.T ∧ s<h.S ∧
    (BitVec.ofInt 64 (BitVec.ofNat 32 pid.row.val).toInt*BitVec.ofNat 64 (h.T*h.S)).toInt=(pid.row.val*(h.T*h.S):Nat) ∧
    (BitVec.ofNat 64 t*BitVec.ofNat 64 h.S).toInt=(t*h.S:Nat) ∧
    (BitVec.ofInt 64 (BitVec.ofNat 32 pid.row.val).toInt*BitVec.ofNat 64 (h.T*h.S)+BitVec.ofNat 64 t*BitVec.ofNat 64 h.S).toInt=
      (pid.row.val*(h.T*h.S)+t*h.S:Nat) ∧
    (offsetWord h pid t s).toInt=(address h pid.row.val t s:Int) ∧
    address h pid.row.val t s<b.input.extent ∧ address h pid.row.val t s<b.output.extent ∧
    (BitVec.ofNat 64 b.input.base+offsetWord h pid t s*BitVec.ofNat 64 b.input.bytes).toNat=
      b.input.base+address h pid.row.val t s*b.input.bytes ∧
    (BitVec.ofNat 64 b.output.base+offsetWord h pid t s*BitVec.ofNat 64 b.output.bytes).toNat=
      b.output.base+address h pid.row.val t s*b.output.bytes

theorem certify (h : Call) (c : Config) (hc : c∈configurations) (b : Backing h) (pid : PID h) :
    ProgramCertificate h c b pid := by
  have positive := (Kernels.configurations_positive c hc).1
  have tb := time_bound h b pid
  have count : chunks h c<signedLimit :=
    (ReversedCumsumScalar.Geometry.chunks_le_time h.timeView c positive).trans_lt tb
  refine ⟨row_widen h pid,feature_widen h pid,signed64 _ count,decrement _ count,
    fun q hq => decrement q (hq.trans count),?_,by decide,?_,?_,?_,?_⟩
  · intro q hq
    rw [BitVec.slt_eq_decide,signed64 q (hq.trans count)]
    have neg : (-1#64).toInt=(-1:Int) := by decide
    simp only [neg,decide_eq_true_eq]
    omega
  · intro q hq
    rw [BitVec.sle_eq_decide,signed64 q (hq.trans count)]
    simp
  · intro q hq i
    rw [time_word]
    exact signed64 _ ((Geometry.time_bound h c positive q hq i).trans tb)
  · intro q hq j
    rw [feature_word]
    exact signed64 _ ((Geometry.feature_bound h pid.feature j).trans (space_bound h c positive b pid q hq))
  · intro q hq i j
    have ts := Geometry.chunk_start h c positive q hq
    have ss := Geometry.feature_start h pid.feature
    have sb := space_bound h c positive b pid q hq
    have ht := Geometry.time_bound h c positive q hq i
    have hs := Geometry.feature_bound h pid.feature j
    have win := windows h b pid _ _ ht hs
    have il : i.val<signedLimit := by
      have := i.isLt
      have := (Kernels.configurations_positive c hc).2
      dsimp only [signedLimit]
      omega
    have jl : j.val<signedLimit := by have := j.isLt; dsimp [BS,signedLimit] at *; omega
    dsimp only
    have offsetBound := win.1.trans (ReversedCumsumScalar.NativeIndex.extent_bound b.input)
    have parts : pid.row.val*(h.T*h.S)<signedLimit ∧ Geometry.time h c q i*h.S<signedLimit ∧
        pid.row.val*(h.T*h.S)+Geometry.time h c q i*h.S<signedLimit := by
      dsimp only [Geometry.address] at offsetBound
      omega
    refine ⟨?_,?_,?_,?_,?_,?_,ht,hs,?_,?_,?_,?_,win.1,win.2,?_,?_⟩
    · rw [←BitVec.ofNat_mul]
      exact signed64 _ (ts.trans tb)
    · rw [←BitVec.ofNat_mul]
      exact signed64 _ (ss.trans sb)
    · rw [BitVec.ofNat_sub_ofNat_of_le h.T (q*c.BT) (by dsimp [signedLimit] at tb; omega) (Nat.le_of_lt ts)]
      exact signed64 _ ((Nat.sub_le _ _).trans_lt tb)
    · rw [BitVec.ofNat_sub_ofNat_of_le h.S (pid.feature.val*BS) (by dsimp [signedLimit] at sb; omega) (Nat.le_of_lt ss)]
      exact signed64 _ ((Nat.sub_le _ _).trans_lt sb)
    · rw [BitVec.slt_eq_decide,signed64 _ il,signed64 _ ((Nat.sub_le _ _).trans_lt tb)]
      simp [activeT]
    · rw [BitVec.slt_eq_decide,signed64 _ jl,signed64 _ ((Nat.sub_le _ _).trans_lt sb)]
      simp [activeS]
    · rw [row_widen,←BitVec.ofNat_mul]
      exact signed64 _ parts.1
    · rw [←BitVec.ofNat_mul]
      exact signed64 _ parts.2.1
    · rw [row_widen,←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_add]
      exact signed64 _ parts.2.2
    · rw [offsetWord_exact]
      exact signed64 _ (win.1.trans (ReversedCumsumScalar.NativeIndex.extent_bound b.input))
    · rw [offsetWord_exact]
      exact pointer_word b.input _ win.1
    · rw [offsetWord_exact]
      exact pointer_word b.output _ win.2
#axiomsClean certify
end
end VeriTile.Bench.Repairs.ReversedCumsum.NativeIndex

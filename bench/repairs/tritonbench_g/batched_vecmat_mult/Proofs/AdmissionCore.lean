import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteIndex
import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostPrepare

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Whole
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open ValueProofs.Host
noncomputable section
/-- Necessary facts of successful ordinary CUDA/Triton setup. An empty grid
does not need a CUDA grid admission; there are no program IDs in that case.
The callback can reject any unsupported block/resource/tuning combination.
No new Python guard or restriction to the two test configurations is imposed. -/
structure LaunchABI (l : Launch) : Prop where
  blockM : l.parameters.BM ≤ 2^31
  blockN : l.parameters.BN ≤ 2^31
  blockK : l.parameters.BK ≤ 2^31
  activeGrid : 0 < ValueProofs.gridCount l.parameters →
    ValueProofs.gridM l.parameters ≤ 2^31 ∧ ValueProofs.gridN l.parameters ≤ 65535
  warps : ∃ e : Nat, l.numWarps = 2^e

/-- Installed Triton 3.7.1 scalar ABI: the value 1 is constexpr; other
nonnegative values below 2^31 use signed i32, and larger values use signed i64.
The source explicitly casts N/K to i64 before any index multiplication. -/
def runtimeDimension (n : Nat) : BitVec 64 :=
  if n = 1 then Index64.word 1 else
  if n < 2^31 then (BitVec.ofNat 32 n).signExtend 64 else Index64.word n

@[simp] theorem runtime_dimension_word (n : Nat) :
    runtimeDimension n = Index64.word n := by
  unfold runtimeDimension
  split
  · simp_all
  · split
    · exact Index64.signed32_cast_word n ‹_›
    · rfl

theorem runtime_dimension_exact (n : Nat) (h : n < 2^63) :
    (runtimeDimension n).toInt = (n : Int) := by
  rw [runtime_dimension_word,Index64.word_toInt h]

theorem signed_loop_bound_exact (K BK : Nat) (hK : K < 2^63) (hBK : BK < 2^63) :
    ((Index64.word K).sdiv (Index64.word BK)).toInt = ((K/BK : Nat) : Int) := by
  have small : ((Index64.word K).sdiv (Index64.word BK)).toNat < 2^63 := by
    rw [Index64.loop_bound_exact K BK hK hBK]
    exact lt_of_le_of_lt (Nat.div_le_self K BK) hK
  rw [BitVec.toInt_eq_toNat_of_lt (by omega),Index64.loop_bound_exact K BK hK hBK]

/-- Every relevant finite-width component is now an explicit conclusion of
successful allocation/setup, rather than an unused isolated lemma. -/
structure IntegerAgreement (p : Parameters) : Prop where
  storage : Index64.StorageLimits p
  addresses : Index64.AddressAgreement p
  metadata : (runtimeDimension p.N).toInt = (p.N : Int) ∧
    (runtimeDimension p.K).toInt = (p.K : Int)
  programIds : ∀ pm : Fin (ValueProofs.gridM p), ∀ pn : Fin (ValueProofs.gridN p),
    (BitVec.ofNat 32 pm.val).signExtend 64 = Index64.word pm.val ∧
    (BitVec.ofNat 32 pn.val).signExtend 64 = Index64.word pn.val
  laneM : ∀ i : Fin p.BM, (BitVec.ofNat 32 i.val).signExtend 64 = Index64.word i.val
  laneN : ∀ j : Fin p.BN, (BitVec.ofNat 32 j.val).signExtend 64 = Index64.word j.val
  laneK : ∀ e : Fin p.BK, (BitVec.ofNat 32 e.val).signExtend 64 = Index64.word e.val
  loopBound : ((Index64.word p.K).sdiv (Index64.word p.BK)).toNat = p.K/p.BK
  loopBoundSigned : ((Index64.word p.K).sdiv (Index64.word p.BK)).toInt =
    ((p.K/p.BK : Nat) : Int)
  loopTest : ∀ c, c ≤ p.K/p.BK →
    ((Index64.word c).toInt < ((Index64.word p.K).sdiv (Index64.word p.BK)).toInt ↔
      c < p.K/p.BK)
  loopStep : ∀ c, c < p.K/p.BK →
    (Index64.word c).toInt = (c : Int) ∧
    (Index64.word c + Index64.word 1).toInt = ((c+1 : Nat) : Int)
  outputBytes : ∀ pm : Fin (ValueProofs.gridM p), ∀ pn : Fin (ValueProofs.gridN p),
    ∀ i : Fin p.BM, ∀ j : Fin p.BN,
      ((pm.val*p.BM+i.val)*p.N+(pn.val*p.BN+j.val))*4+3 < p.M*p.N*4
  inputBytes : ∀ pm : Fin (ValueProofs.gridM p), ∀ pn : Fin (ValueProofs.gridN p),
    ∀ c : Fin (p.K/p.BK), ∀ i : Fin p.BM, ∀ j : Fin p.BN, ∀ e : Fin p.BK,
      ((pm.val*p.BM+i.val)*p.K+(c.val*p.BK+e.val))*4+3 < p.M*p.K*4 ∧
      ((pm.val*p.BM+i.val)*p.N*p.K+(pn.val*p.BN+j.val)*p.K+
        (c.val*p.BK+e.val))*4+3 < p.M*p.N*p.K*4
  pointerWindow : ∀ base count offset, base+count*4 ≤ 2^64 → offset < count →
    (Index64.bytePointer base (Index64.word offset)).toNat = base+offset*4 ∧
    base ≤ (Index64.bytePointer base (Index64.word offset)).toNat ∧
    (Index64.bytePointer base (Index64.word offset)).toNat+3 < base+count*4

theorem launch_integer_agreement (l : Launch) (storage : Index64.StorageLimits l.parameters)
    (native : LaunchABI l) : IntegerAgreement l.parameters where
  storage := storage
  addresses := Index64.complete_address_agreement l.parameters storage
  metadata := ⟨runtime_dimension_exact _ storage.N,runtime_dimension_exact _ storage.K⟩
  programIds pm pn := by
    have active : 0 < ValueProofs.gridCount l.parameters :=
      Nat.mul_pos (Nat.zero_lt_of_lt pm.isLt) (Nat.zero_lt_of_lt pn.isLt)
    obtain ⟨hm,hn⟩ := native.activeGrid active
    exact ⟨Index64.signed32_cast_word _ (lt_of_lt_of_le pm.isLt hm),
      Index64.signed32_cast_word _ (by have := pn.isLt; omega)⟩
  laneM i := Index64.arange_cast_word _ i native.blockM
  laneN j := Index64.arange_cast_word _ j native.blockN
  laneK e := Index64.arange_cast_word _ e native.blockK
  loopBound := Index64.loop_bound_exact _ _ storage.K (by have := native.blockK; omega)
  loopBoundSigned := signed_loop_bound_exact _ _ storage.K (by have := native.blockK; omega)
  loopTest c hc := by
    have hc64 := lt_of_le_of_lt (le_trans hc (Nat.div_le_self _ _)) storage.K
    rw [Index64.word_toInt hc64,
      signed_loop_bound_exact _ _ storage.K (by have := native.blockK; omega)]
    exact_mod_cast (Iff.rfl : c < l.parameters.K/l.parameters.BK ↔
      c < l.parameters.K/l.parameters.BK)
  loopStep c hc := Index64.loop_increment_exact _ _ _ storage.K hc
  outputBytes pm pn i j := (Index64.all_output_addresses l.parameters storage pm pn i j).2
  inputBytes pm pn c i j e := (Index64.all_input_addresses l.parameters storage pm pn c i j e).2.2
  pointerWindow := Index64.byte_pointer_exact

#axiomsClean runtime_dimension_word
#axiomsClean runtime_dimension_exact
#axiomsClean signed_loop_bound_exact
#axiomsClean launch_integer_agreement
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.Whole

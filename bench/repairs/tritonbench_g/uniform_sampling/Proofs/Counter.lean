import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Model
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Spec
import bench.tritonbench_g.uniform_sampling.Proofs.TypedCounterOperands
import bench.tritonbench_g.uniform_sampling.Proofs.TypedTuple

namespace VeriTile.Bench.Repairs.UniformSampling.CounterProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.UniformSampling

 theorem signed_range {x : Nat} (hx:x<2^64) :
    Random.IntegerType.i64.inRange (signed64 x)=Bool.true := by
  simp only [signed64,Random.IntegerType.inRange,decide_eq_true_eq]
  split <;> omega

theorem signed_bits (x : Nat) : BitVec.ofInt 64 (signed64 x)=BitVec.ofNat 64 x := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofInt,BitVec.toNat_ofNat]
  unfold signed64
  split
  · norm_cast
  · rw [Int.sub_emod]
    norm_num
    norm_cast

theorem signed_low (x : Nat) : BitVec.ofInt 32 (signed64 x)=BitVec.ofNat 32 x := by
  have h:=congrArg (BitVec.setWidth 32) (signed_bits x)
  simpa only [CounterOperands.truncate_ofInt (by decide : 32≤64),
    BitVec.setWidth_ofNat_of_le (by decide : 32≤64)] using h

theorem signed_high (x : Nat) :
    ScalarSpec.counterHigh (signed64 x)=BitVec.ofNat 32 (x/2^32) := by
  unfold ScalarSpec.counterHigh signed64
  split
  · change BitVec.ofInt 32 ((x : Int)/(2^32 : Int)) = _
    norm_cast
  · have h : ((x : Int)-2^64)/2^32=(x : Int)/2^32-2^32 := by omega
    rw [h]
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_ofInt,BitVec.toNat_ofNat]
    rw [Int.sub_emod]
    norm_num
    norm_cast

theorem seed_key (x : Nat) : Random.seedKey (signed64 x)=Random.seedKey (x : Int) := by
  simp only [Random.seedKey,signed_bits,BitVec.ofInt_natCast]

theorem no_carry {start index : Nat} (h:start%2^32+index<2^32) :
    ScalarSpec.counter (signed64 start) index=Spec.counter (start+index) := by
  have hd : (start+index)/2^32=start/2^32 := by omega
  simp only [ScalarSpec.counter,ScalarSpec.counterLow,Spec.counter,signed_low,
    BitVec.ofNat_add_ofNat,signed_high,hd]

theorem words {seed start index : Nat} (h:start%2^32+index<2^32) :
    ScalarSpec.words (signed64 seed) (signed64 start) index=Spec.words seed (start+index) := by
  simp only [ScalarSpec.words,Spec.words,no_carry h,seed_key]

/-- The original typed tuple proof applies to the repaired, widened integer
prefix once its registers are prepared. This reuses all five snapshots, all
four actual Philox field evaluations, and the fixed ten-round integer engine. -/
theorem typed_tuple {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (B seed start : Nat) (index : TileIndex [B]→Nat) (s : FP.Structural.State α)
    (hseed:seed<2^64)
    (hs:s.regs .int [] "philox_seed"=some (fun _=>signed64 seed))
    (h0:s.regs .nat [B] "c0"=some (fun i=>(ScalarSpec.counterLow (signed64 start) (index i)).toNat))
    (h1:s.regs .nat [] "c1"=some (fun _=>(ScalarSpec.counterHigh (signed64 start)).toNat))
    (hz:s.regs .nat [B] "_O"=some (fun _=>0)) :
    FP.Structural.step M (TypedTuple.tupleBlock B) s=
      some (TypedTuple.outputState (signed64 seed) (signed64 start) index
        (TypedTuple.capturedState (signed64 seed) (signed64 start) index s)) := by
  exact TypedTuple.tuple_exec M (signed64 seed) (signed64 start) index s
    (signed_range hseed) hs h0 h1 hz

#axiomsClean signed_range
#axiomsClean signed_bits
#axiomsClean signed_high
#axiomsClean no_carry
#axiomsClean words
#axiomsClean typed_tuple
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.UniformSampling.CounterProof

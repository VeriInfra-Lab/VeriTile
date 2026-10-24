import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.PhysicalBindings

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Native
noncomputable section

/-- An unannotated Python ignore integer may select signed or unsigned i64.
Both preserve these bits. Labels still come from actual signed-int64 storage. -/
def ignoredMatches (target ignored : Int) : Bool :=
  decide (BitVec.ofInt 64 target=BitVec.ofInt 64 ignored)

theorem ignored_comparison (target ignored total : Int) (totalBound : 0<total ∧ total<2^63)
    (ignoredBound : -(2^63 : Int) ≤ ignored ∧ ignored<2^64)
    (valid : target=ignored ∨ (0≤target ∧ target<total)) :
    ignoredMatches target ignored=decide (target=ignored) := by
  classical
  unfold ignoredMatches
  rw [decide_eq_decide]
  constructor
  · intro words
    rcases valid with same|admitted
    · exact same
    · by_cases negative : ignored<0
      · have tx : (BitVec.ofInt 64 target).toInt=target := by
          apply BitVec.toInt_ofInt_eq_self <;> norm_num at totalBound ⊢ <;> omega
        have ix : (BitVec.ofInt 64 ignored).toInt=ignored := by
          apply BitVec.toInt_ofInt_eq_self <;> norm_num at ignoredBound ⊢ <;> omega
        have eq := congrArg BitVec.toInt words
        rwa [tx,ix] at eq
      · have nonneg : 0 ≤ ignored := by omega
        have upper : target<2^64 := by norm_num at totalBound ⊢; omega
        have eq := congrArg BitVec.toNat words
        simp only [BitVec.toNat_ofInt,Int.emod_eq_of_lt admitted.1 upper,
          Int.emod_eq_of_lt nonneg ignoredBound.2] at eq
        omega
  · intro same
    rw [same]

/-- The ordinary annotated signed-i64 binder rejects an oversized class
count before launch. This is an API outcome, not a new numerical domain. -/
theorem annotated_class_failure (classes : Nat) (oversized : 2^63≤classes) :
    CrossEntropy1.Physical.scalar64 (classes : Int)=.error "signed int64 conversion" := by
  apply CrossEntropy1.Physical.scalar64_failure
  norm_num at oversized ⊢
  omega

/-- A nonignored label and the selected rank's class start are both below
the actual signed-i64 total. Their difference cannot wrap. -/
theorem relative_label_exact (target start total : Int)
    (totalBound : 0<total ∧ total<2^63) (targetBound : 0≤target ∧ target<total)
    (startBound : 0≤start ∧ start<total) :
    (BitVec.ofInt 64 target-BitVec.ofInt 64 start).toInt=target-start := by
  have targetWord : (BitVec.ofInt 64 target).toInt=target := by
    apply BitVec.toInt_ofInt_eq_self <;> norm_num at totalBound ⊢ <;> omega
  have startWord : (BitVec.ofInt 64 start).toInt=start := by
    apply BitVec.toInt_ofInt_eq_self <;> norm_num at totalBound ⊢ <;> omega
  rw [BitVec.toInt_sub,targetWord,startWord]
  apply Int.bmod_eq_of_le <;> norm_num at totalBound ⊢ <;> omega

/-- The ignored branch selects class_start before subtracting, including
INT64_MIN. No ignored-label subtraction is evaluated in the proof model. -/
theorem guarded_relative_label (target ignored start total : Int)
    (totalBound : 0<total ∧ total<2^63)
    (ignoredBound : -(2^63 : Int) ≤ ignored ∧ ignored<2^64)
    (startBound : 0≤start ∧ start<total)
    (valid : target=ignored ∨ (0≤target ∧ target<total)) :
    ((if ignoredMatches target ignored then BitVec.ofInt 64 start else BitVec.ofInt 64 target)-
      BitVec.ofInt 64 start).toInt=(if target=ignored then 0 else target-start) := by
  rw [ignored_comparison target ignored total totalBound ignoredBound valid]
  by_cases same : target=ignored
  · simp only [same,decide_true,↓reduceIte,BitVec.sub_self,BitVec.toInt_zero]
  · simp only [same,decide_false,Bool.false_eq_true,↓reduceIte]
    exact relative_label_exact target start total totalBound (valid.resolve_left same) startBound

theorem rank_class_metadata (world rank cols : Nat) (positive : 0<cols) (rankBound : rank<world)
    (totalBound : world*cols<2^63) :
    rank*cols<world*cols ∧ rank*cols<2^63 ∧ cols<2^63 := by
  have start := Nat.mul_lt_mul_of_pos_right rankBound positive
  have one : 1≤world := by omega
  have localWidth := Nat.mul_le_mul_right cols one
  simp only [Nat.one_mul] at localWidth
  omega

#axiomsClean ignored_comparison
#axiomsClean annotated_class_failure
#axiomsClean relative_label_exact
#axiomsClean guarded_relative_label
#axiomsClean rank_class_metadata
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Native

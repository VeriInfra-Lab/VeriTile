import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Lowering

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open Source
noncomputable section

/-- The literal and signed-i64 conversion of one have the same binary32 bits.
This is a conversion port, independent of any task expression. -/
def ConstantOne (n : Numerics) : Prop := n.natural 1=n.literal 1

/-- The 32 backward correction FMA sites in the five pinned profiles use
this exact input graph. Some register assignments commute the FMA operands;
the scalar all-pattern commutation obligation is kept explicit. -/
theorem backward_correction_site (profile : ReductionProfile) (n : Numerics) (laws : Peepholes n)
    (scale lse probability : Word) (commuted : Bool) :
    word profile n (.binary .add (.raw probability)
      (.binary .mul (.binary .mul (.binary .mul (.literal 2) (.raw scale)) (.raw lse)) (.raw probability)))=
    if commuted then n.fma probability (n.mul (n.add scale scale) lse) probability
      else n.fma (n.mul (n.add scale scale) lse) probability probability := by
  change n.fma (n.mul (n.mul (n.literal 2) scale) lse) probability probability=_
  rw [laws.twice]
  cases commuted
  · rfl
  · exact laws.fmaComm _ _ _

/-- The forward squared-LSE add contracts only its final multiply/add. The
first product remains rounded and can appear in either FMA operand register. -/
theorem squared_lse_site (profile : ReductionProfile) (n : Numerics) (laws : Peepholes n) (scale lse loss : Word) (commuted : Bool) :
    word profile n (.binary .add (.raw loss) (.binary .mul (.binary .mul (.raw scale) (.raw lse)) (.raw lse)))=
    if commuted then n.fma lse (n.mul scale lse) loss else n.fma (n.mul scale lse) lse loss := by
  change n.fma (n.mul scale lse) lse loss=_
  cases commuted
  · rfl
  · exact laws.fmaComm _ _ _

/-- Forward smoothed-target subtraction uses a separately rounded `1-s`,
its sign-bit negation, and one FMA with the loaded target and previous loss. -/
theorem smoothed_target_site (profile : ReductionProfile) (n : Numerics) (one : ConstantOne n) (smoothing target previous : Word) :
    word profile n (.binary .sub (.raw previous)
      (.binary .mul (.binary .sub (.natural 1) (.raw smoothing)) (.raw target)))=
      n.fma (n.neg (n.sub (n.literal 1) smoothing)) target previous := by
  change n.fma (n.neg (n.sub (n.natural 1) smoothing)) target previous=_
  rw [one]

theorem subtract_one_site (profile : ReductionProfile) (n : Numerics) (laws : Peepholes n) (x : Word) :
    word profile n (.binary .sub (.raw x) (.literal 1))=n.add x (n.literal (-1)) := laws.minusOne x

/-- Elementary exp lowering uses this exact native log2(e) bit constant.
The numerical port remains a single exp operation, not a row/kernel oracle. -/
def ExpImplementation (n : Numerics) (exp2Approx : Word → Word) : Prop :=
  ∀ x, n.exp x=exp2Approx (n.mul x 0x3fb8aa3b)

theorem exp_site (profile : ReductionProfile) (n : Numerics) (exp2Approx : Word → Word) (implementation : ExpImplementation n exp2Approx)
    (x : Word) : word profile n (.unary .exp (.raw x))=exp2Approx (n.mul x 0x3fb8aa3b) := implementation x

#axiomsClean backward_correction_site
#axiomsClean squared_lse_site
#axiomsClean smoothed_target_site
#axiomsClean subtract_one_site
#axiomsClean exp_site
#specNonCircular ConstantOne avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular ExpImplementation avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating

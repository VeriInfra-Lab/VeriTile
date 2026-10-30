import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Lowering

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source
noncomputable section

/-- The literal and signed-i64 conversion of one have the same binary32 bits.
This is a conversion port, independent of any task expression. -/
def ConstantOne (n : Ports) : Prop := n.natural 1=n.literal 1

/-- Backward correction FMA sites use this exact input graph. Some register assignments commute the FMA operands;
the scalar all-pattern commutation obligation is kept explicit. -/
theorem backward_correction_site (n : Ports) (profile : Profile) (laws : Peepholes n)
    (scale lse probability : Word) (commuted : Bool) :
    word n profile (.binary .add (.raw probability)
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
theorem squared_lse_site (n : Ports) (profile : Profile) (laws : Peepholes n) (scale lse loss : Word) (commuted : Bool) :
    word n profile (.binary .add (.raw loss) (.binary .mul (.binary .mul (.raw scale) (.raw lse)) (.raw lse)))=
    if commuted then n.fma lse (n.mul scale lse) loss else n.fma (n.mul scale lse) lse loss := by
  change n.fma (n.mul scale lse) lse loss=_
  cases commuted
  · rfl
  · exact laws.fmaComm _ _ _

/-- Forward smoothed-target subtraction uses a separately rounded `1-s`,
its sign-bit negation, and one FMA with the loaded target and previous loss. -/
theorem smoothed_target_site (n : Ports) (profile : Profile) (one : ConstantOne n) (smoothing target previous : Word) :
    word n profile (.binary .sub (.raw previous)
      (.binary .mul (.binary .sub (.natural 1) (.raw smoothing)) (.raw target)))=
      n.fma (n.neg (n.sub (n.literal 1) smoothing)) target previous := by
  change n.fma (n.neg (n.sub (n.natural 1) smoothing)) target previous=_
  rw [one]

theorem subtract_one_site (n : Ports) (profile : Profile) (laws : Peepholes n) (x : Word) :
    word n profile (.binary .sub (.raw x) (.literal 1))=n.add x (n.literal (-1)) := laws.minusOne x

/-- Elementary exp lowering uses this exact native log2(e) bit constant.
The numerical port remains a single exp operation, not a row/kernel oracle. -/
def ExpImplementation (n : Ports) (exp2Approx : Word → Word) : Prop :=
  ∀ x, n.exp x=exp2Approx (n.mul x 0x3fb8aa3b)

theorem exp_site (n : Ports) (profile : Profile) (exp2Approx : Word → Word) (implementation : ExpImplementation n exp2Approx)
    (x : Word) : word n profile (.unary .exp (.raw x))=exp2Approx (n.mul x 0x3fb8aa3b) := implementation x


/-- The scaled target subtraction can negate either factor before the FMA.
This is a separate scalar obligation, including NaN/sign-bit behavior. -/
theorem scaled_target_site (n : Ports) (profile : Profile) (laws : Peepholes n)
    (target scale loss : Word) :
    word n profile (.binary .sub (.raw loss) (.binary .mul (.raw target) (.raw scale)))=
      n.fma (n.neg scale) target loss := laws.negFmaExchange target scale loss

theorem scaled_logit_site (n : Ports) (profile : Profile) (laws : Peepholes n) (target scale : Word) :
    word n profile (.binary .mul (.raw target) (.raw scale))=n.mul scale target := laws.mulComm target scale

theorem saved_z_site (n : Ports) (profile : Profile) (laws : Peepholes n) (scale lse : Word) :
    word n profile (.binary .mul (.binary .mul (.raw scale) (.raw lse)) (.raw lse))=
      n.mul lse (n.mul scale lse) := laws.mulComm _ _

theorem stable_lse_site (n : Ports) (profile : Profile) (laws : Peepholes n) (logsum maximum : Word) :
    word n profile (.binary .add (.raw logsum) (.raw maximum))=n.add maximum logsum := laws.addComm _ _

theorem host_scale_site (n : Ports) (profile : Profile) (scale : ℝ) (lse : Word) :
    word n profile (.hostScale scale (.raw lse))=n.hostMul (n.literal scale) lse := rfl

#axiomsClean scaled_target_site
#axiomsClean scaled_logit_site
#axiomsClean saved_z_site
#axiomsClean stable_lse_site
#axiomsClean host_scale_site

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
end VeriTile.Bench.Repairs.CrossEntropy2.Floating

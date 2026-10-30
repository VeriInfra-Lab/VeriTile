import VeriTile.Triton.Random.Philox
import VeriTile.Meta.StatementAudit

/-! A numerical obligation for the pinned multinomial source. Its rounded
`cum_scores - scores` need not recover the preceding accumulator. This is a
concrete scalar boundary proof, not a whole-kernel execution or native theorem. -/
namespace VeriTile.Tests.MultinomialRoundedBoundary
open VeriTile.Triton

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

def sample : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3d63666a
def nextWeight : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3e800000
def cumulative : FP.Value .fp32 := FP.add FP.Config.ieeeValue sample nextWeight
def preceding : FP.Value .fp32 := FP.sub FP.Config.ieeeValue cumulative nextWeight

theorem actual_rng_sample : (Random.rand 0 1 .bits32).raw = sample.raw := by decide +kernel

theorem rounded_sum_bits : cumulative.raw = 0x3e9c6ccd := by decide +kernel

theorem rounded_preceding_bits : preceding.raw = 0x3d636668 := by decide +kernel

/-- The source's interval in the next block can overlap the prior boundary. -/
theorem overlapping_boundary :
    FP.lt FP.Config.ieeeValue preceding sample = Bool.true ∧
    FP.lt FP.Config.ieeeValue sample cumulative = Bool.true := by decide +kernel

theorem sample_positive :
    FP.lt FP.Config.ieeeValue (FP.Value.ofBits .fp32 0) sample = Bool.true := by decide +kernel

theorem preceding_not_previous_accumulator : preceding ≠ sample := by decide +kernel

#axiomsClean actual_rng_sample
#axiomsClean rounded_sum_bits
#axiomsClean rounded_preceding_bits
#axiomsClean overlapping_boundary
#axiomsClean sample_positive
#axiomsClean preceding_not_previous_accumulator
end VeriTile.Tests.MultinomialRoundedBoundary

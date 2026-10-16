import bench.tritonbench_g.layernorm_fwd_triton.FPCorrect
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages (literal32)
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 22000

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessSmall : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000
def witnessThree : FP.Value .fp32 := FP.Value.ofBits .fp32 0x40400000
def witnessMean : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f2aaaab

def witnessParameters : Parameters where
  x := "x"
  weight := "w"
  y := "y"
  xSeqStride := 17
  xHeadStride := 5
  xDimStride := 19
  ySeqStride := 17
  yHeadStride := 5
  yDimStride := 23
  weightHeadStride := 5
  weightDimStride := 29
  columns := 3
  block := 2
  epsilon := 0

def witnessState : State (FP.Value .fp32) where
  mem := fun r o => .mk .fp32 (if r=("x" : RegionName) ∧ o=1 then witnessSmall else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem literal_three : literal32 (fun _ => witnessThree) (3 : ℝ)=witnessThree := by
  norm_num [literal32]

/-- N=3, B=2 uses two iterations of each of the three loops. The tiny
positive input is lost in a rounded addition, then division rounds again. -/
theorem witness_mean_rounding :
    bitMean .ieeeValue (fun _ => witnessThree) witnessParameters witnessState=witnessMean := by
  unfold bitMean
  change FP.div .ieeeValue _ (literal32 (fun _ => witnessThree) (3 : ℝ))=witnessMean
  rw [literal_three]
  decide +kernel

theorem witness_not_exact : witnessSmall.toRat?=some (1/33554432) ∧
    witnessMean.toRat?=some (11184811/16777216) ∧
    ((2+1/33554432)/3 : Rat)≠11184811/16777216 := by decide +kernel

/-- Successful nonempty complete source execution, all three output cells,
a protected tail cell, and a checked intermediate rounding event. sqrt stays
an explicit primitive rather than a fabricated hardware implementation. -/
theorem full_stream_nonempty_rounding (rest : Algebra (FP.Value .fp32))
    (sqrt32 : FP.Value .fp32 → FP.Value .fp32) :
    ∃ t, FP.Structural.exec (software rest .ieeeValue (fun _ => witnessThree) sqrt32)
      (kernel witnessParameters) witnessState=some t ∧
      (∀ n, n<3 → t.mem "y" n=.mk .fp32
        (bitY .ieeeValue (fun _ => witnessThree) sqrt32 witnessParameters witnessState n)) ∧
      t.mem "y" 3=witnessState.mem "y" 3 ∧
      bitMean .ieeeValue (fun _ => witnessThree) witnessParameters witnessState=witnessMean ∧
      ((2+1/33554432)/3 : Rat)≠11184811/16777216 := by
  obtain ⟨t,he,hy,hf⟩ := FPCorrect.fp32_scalarops_full_stream_correctness rest .ieeeValue
    (fun _ => witnessThree) sqrt32 witnessParameters witnessState (by decide) (by decide) (by decide)
  refine ⟨t,he,?_,?_,witness_mean_rounding,witness_not_exact.2.2⟩
  · simpa [yBase, witnessParameters, witnessState] using hy
  · apply hf
    right
    intro n hn
    simp [yBase, witnessParameters, witnessState]
    change n<3 at hn
    omega

#axiomsClean literal_three
#axiomsClean witness_mean_rounding
#axiomsClean witness_not_exact
#axiomsClean full_stream_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages

import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Grid
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.KernelArithmetic
open VeriTile.Triton FP.Structural
noncomputable section
abbrev FPWord := FP.Value .fp32
instance : Inhabited FPWord := ⟨FP.Value.zero .fp32⟩

/-- Independent mathematical target, in the source operand order. -/
def mathematical (x : ℝ) : ℝ := x*(2:ℝ)^112
def constant : FPWord := ⟨0x77800000⟩

/-- The selected CUDA multiplication model: nearest-even and no input/output
FTZ. Offline lowering evidence is recorded separately; this definition is not
a theorem about a GPU implementation. Other explicit profiles are also covered. -/
def selectedProfile : FP.Config := FP.Config.ieeeValue
def native (profile : FP.Config) (x : FPWord) : FPWord := FP.mul profile x constant

def realAlgebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | .max => max x y | .pow => 0
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

/-- Only the source's pinned constexpr reaches this literal interpretation.
The proof below fixes its exact FP32 bit pattern and mathematical value. -/
def literalWord (x : ℝ) : FPWord :=
  if x=exponentCompensator then constant else FP.Value.zero .fp32

def nativeAlgebra (profile : FP.Config) : Algebra FPWord where
  literal := fun _ _ x => literalWord x
  negInf := FP.Value.infinity .fp32 Bool.true
  binary := fun _ _ op x y => match op with
    | .add => FP.add profile x y | .sub => FP.sub profile x y
    | .mul => FP.mul profile x y | .div => FP.div profile x y
    | _ => FP.Value.zero .fp32
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => FP.round .fp32 profile n
  fromInt := fun _ n => FP.round .fp32 profile n
  fp32Bits := fun b => ⟨b.bits⟩
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => FP.Value.zero .fp32
  reduceSum := fun _ _ _ _ _ _ => FP.Value.zero .fp32

theorem scalar_real (x : ℝ) : scalar realAlgebra x=mathematical x := rfl
theorem scalar_native (profile : FP.Config) (x : FPWord) :
    scalar (nativeAlgebra profile) x=native profile x := by
  simp [scalar,nativeAlgebra,literalWord,native]

theorem constant_exact : constant.toRat?=some ((2:Rat)^112) := by
  have raw : constant.raw=2004877312 := rfl
  norm_num [FP.Value.toRat?,FP.Value.decode,FP.Value.exponent,FP.Value.fraction,
    FP.Value.sign,raw,FP.Format.fractionBits,FP.pow2,Int.toNat]

/-- Complete selected typed DSL for one program, with every masked cell
framed. No success or final-value premise is assumed. -/
theorem program (M : Algebra α) [Inhabited α] (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec M (candidateKernel p.x p.output p.n p.block) s=some final ∧
      ∀ r i, final.mem r i=
        if r = p.output ∧ s.pids 0 * p.block ≤ i ∧ i < s.pids 0 * p.block + p.block ∧ i < p.n
        then resultCell M p s.mem i else s.mem r i := by
  obtain ⟨final,he,hm⟩ := complete_correct M p s
  exact ⟨final,he,fun r i => (congrFun (congrFun hm r) i).trans (expected_cell M p s r i)⟩

theorem real_grid (p : Parameters) (s : State ℝ) (positive : 0<p.block) :
    ∃ final,runGrid realAlgebra p s=some final ∧
      ∀ r i,final.mem r i=if r=p.output ∧ i<p.n then
        Cell.mk .real (mathematical ((s.mem p.x i).read .real)) else s.mem r i := by
  obtain ⟨final,he,hm⟩ := grid_correct realAlgebra p s positive
  refine ⟨final,he,?_⟩
  intro r i
  rw [hm]
  rfl

theorem native_grid (profile : FP.Config) (p : Parameters) (s : State FPWord)
    (positive : 0<p.block) :
    ∃ final,runGrid (nativeAlgebra profile) p s=some final ∧
      ∀ r i,final.mem r i=if r=p.output ∧ i<p.n then
        Cell.mk .real (native profile ((s.mem p.x i).read .real)) else s.mem r i := by
  obtain ⟨final,he,hm⟩ := grid_correct (nativeAlgebra profile) p s positive
  refine ⟨final,he,?_⟩
  intro r i
  rw [hm]
  simp only [gridExpectedMemory,resultCell,scalar_native]

#specNonCircular mathematical avoiding [candidateKernel,sourceBody,runGrid,FP.Structural.exec]
#specNonCircular native avoiding [candidateKernel,sourceBody,runGrid,FP.Structural.exec]
#axiomsClean constant_exact
#axiomsClean program
#axiomsClean real_grid
#axiomsClean native_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.KernelArithmetic

import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Program

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false

abbrev Word := FP.Value .fp32
abbrev Byte := BitVec 8
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Ordinary backend meanings of exactly the three FP32 source primitives.
These are functions, not numerical equivalence assumptions or experiments. -/
structure Primitives where
  divide : Word → Word → Word
  logarithm : Word → Word
  multiply : Word → Word → Word

/-- Independent ordered native graph on logical input words. -/
def reference (p : Primitives) (x y : Word) : Word :=
  p.multiply x (p.logarithm (p.divide x y))

/-- Interpret the typed source in its selected precision. The source contains
no other floating operation; unused Algebra fields cannot affect the theorem. -/
def algebra (p : Primitives) : Algebra Word where
  literal := fun _ _ _ => FP.Value.zero .fp32
  negInf := FP.Value.zero .fp32
  binary := fun _ _ op a b => match op with
    | .div => p.divide a b
    | .mul => p.multiply a b
    | _ => FP.Value.zero .fp32
  unary := fun _ op a => match op with
    | .log => p.logarithm a
    | _ => FP.Value.zero .fp32
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => FP.Value.zero .fp32
  fromInt := fun _ _ => FP.Value.zero .fp32
  fp32Bits := fun bits => ⟨bits.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => FP.Value.zero .fp32
  reduceSum := fun _ {_} _ _ _ _ => FP.Value.zero .fp32

theorem scalar_reference (p : Primitives) (x y : Word) :
    Program.scalar (algebra p) x y = reference p x y := rfl

/-- Complete raw typed-source execution with the independent FP32
 divide/log/multiply graph and an exact all-cell memory frame. -/
theorem kernel_execution (ops : Primitives) (p : Parameters) (initial : State Word)
    (positive : 0 < p.block) :
    ∃ final, Program.grid (algebra ops) p initial = some final ∧
      final.mem = fun r i => if r = p.output ∧ i < p.n then
        .mk .real (reference ops ((initial.mem p.x i).read .real)
          ((initial.mem p.y i).read .real)) else initial.mem r i := by
  obtain ⟨final, executed, memory⟩ := Program.grid_complete (algebra ops) p initial positive
  exact ⟨final, executed, by simpa only [Program.resultCell, scalar_reference,
    VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory] using memory⟩

#specNonCircular reference avoiding [KernelDSL.kernel,Program.scalar,Program.grid]
#specNonCircular Primitives avoiding [KernelDSL.kernel,reference,Program.grid]
#axiomsClean scalar_reference
#axiomsClean kernel_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.FPEquiv

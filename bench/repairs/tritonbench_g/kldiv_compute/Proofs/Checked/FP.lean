import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Checked.Host
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Program
import bench.tritonbench_g.kldiv_compute.Proofs.Precision
import bench.tritonbench_g.kldiv_compute.Proofs.Source

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous.FPProof
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Word := FP.Value .fp32

structure Numerics where
  divide : Word → Word → Word
  logarithm : Word → Word
  multiply : Word → Word → Word

/-- The unchanged source order: FP32 division, tl.log, then multiplication.
There is no algebraic log rewrite, zero extension, or precision promotion. -/
def formula (n : Numerics) (x y : Word) : Word := n.multiply x (n.logarithm (n.divide x y))

def algebra (rest : Algebra Word) (n : Numerics) : Algebra Word :=
  {rest with
    fp32Load := id
    binary := fun precision dtype op x y => match op with
      | .div => n.divide x y
      | .mul => n.multiply x y
      | _ => rest.binary precision dtype op x y
    unary := fun precision op x => match op with
      | .log => n.logarithm x
      | _ => rest.unary precision op x}

theorem source_formula (rest : Algebra Word) (n : Numerics) (x y : Word) :
    Program.scalar (algebra rest n) x y=formula n x y := rfl

/-- The finite interpreter's exact AST is the erasure of the selected FP32
source; the typed scalar theorem above separately retains its FP32 operations. -/
theorem typed_ast_lowering (n : Nat) :
    Source.lowerBody (source ⟨"x","y","out",n,1024⟩).eraseDType.toAlgKernel.body=
      some (Source.sourceCode "x" "y" "out" n) := by
  rw [source,VeriTile.Bench.TritonBenchG.KldivCompute.typed_projection]
  exact Source.original_ast_lowering "x" "y" "out" n

structure Domain where
  divide : Word → Word → Prop
  logarithm : Word → Prop
  multiply : Word → Word → Prop

/-- Only fixed-arity local scalar equalities, at the actual native operands. -/
structure LocalPorts (native reference : Numerics) (domain : Domain) : Prop where
  divide : ∀ x y,domain.divide x y → native.divide x y=reference.divide x y
  logarithm : ∀ x,domain.logarithm x → native.logarithm x=reference.logarithm x
  multiply : ∀ x y,domain.multiply x y → native.multiply x y=reference.multiply x y

def Admitted (native : Numerics) (domain : Domain) (x y : Word) : Prop :=
  domain.divide x y ∧ domain.logarithm (native.divide x y) ∧
    domain.multiply x (native.logarithm (native.divide x y))

theorem transport (native reference : Numerics) (domain : Domain) (ports : LocalPorts native reference domain)
    (x y : Word) (admitted : Admitted native domain x y) : formula native x y=formula reference x y := by
  unfold formula
  rw [ports.multiply _ _ admitted.2.2,ports.logarithm _ admitted.2.1,ports.divide _ _ admitted.1]

#axiomsClean source_formula
#axiomsClean typed_ast_lowering
#axiomsClean transport
#specNonCircular formula avoiding [Program.grid,Source.runProgram,Binding.runWrapper]
#specNonCircular LocalPorts avoiding [Program.grid,Source.runProgram,Binding.runWrapper,formula]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Contiguous.FPProof

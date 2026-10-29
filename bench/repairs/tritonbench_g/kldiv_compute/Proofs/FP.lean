import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Storage

namespace VeriTile.Bench.Repairs.KldivCompute.Floating
open VeriTile.Triton FP.Structural
open Physical (Word)
noncomputable section

/-- Actual production FP32 primitive ports. No Real arithmetic or rounding law
is imposed. Each port accepts every raw binary32 pattern, including NaNs,
infinities, signed zero and subnormals. Compiler configuration is fixed by the
SM90/default Triton compilation evidence; portability remains explicit. -/
structure Numerics where
  div : Word → Word → Word
  log : Word → Word
  mul : Word → Word → Word

def backend (rest : Algebra Word) (n : Numerics) : Algebra Word :=
  { rest with
    fp32Load := id
    binary := fun p d op a b => match p,d,op with
      | some .fp32,.real,.div => n.div a b
      | some .fp32,.real,.mul => n.mul a b
      | _,_,_ => rest.binary p d op a b
    unary := fun p op a => match p,op with
      | some .fp32,.log => n.log a
      | _,_ => rest.unary p op a }

def formula (n : Numerics) (x y : Word) := n.mul x (n.log (n.div x y))
theorem source_formula (rest : Algebra Word) (n : Numerics) (x y : Word) :
    Program.scalar (backend rest n) x y=formula n x y := rfl

/-- Independent selected target algebra; only these three reached operations
are compared. FP32 loads preserve raw words as a storage law. -/
def reference (target : Algebra Word) (x y : Word) : Word :=
  target.binary (some .fp32) .real .mul x
    (target.unary (some .fp32) .log (target.binary (some .fp32) .real .div x y))

structure Obligations (n : Numerics) (target : Algebra Word) : Prop where
  div : ∀ x y, n.div x y=target.binary (some .fp32) .real .div x y
  log : ∀ x, n.log x=target.unary (some .fp32) .log x
  mul : ∀ x y, n.mul x y=target.binary (some .fp32) .real .mul x y

theorem scalar_equiv (rest : Algebra Word) (n : Numerics) (target : Algebra Word)
    (h : Obligations n target) (x y : Word) :
    Program.scalar (backend rest n) x y=reference target x y := by
  simp only [source_formula,formula,reference,h.div,h.log,h.mul]

/-- Original helper reused: the unchanged floating suffix remains exactly
equivalent when only the natural index multiplication is commuted. This is
additional structural evidence, not a numerical acceptance substitute. -/
theorem original_structural (rest : Algebra Word) (n : Numerics)
    (p : Parameters) (s : State Word) :
    run (FP.ProfiledRewrite.engine (backend rest n)) (source p).surfaceBody s =
    run (FP.ProfiledRewrite.engine (backend rest n))
      (VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel
        .fp32 .fp32 .fp32 Bool.true p.x p.y p.output p.n p.block).surfaceBody s :=
  VeriTile.Bench.TritonBenchG.KldivCompute.FPProof.source_equivalence
    (backend rest n) .fp32 .fp32 .fp32 p.x p.y p.output p.n p.block s

#specNonCircular reference avoiding [backend,Program.scalar,Program.grid,Host.run]
#axiomsClean scalar_equiv
#axiomsClean original_structural
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Floating

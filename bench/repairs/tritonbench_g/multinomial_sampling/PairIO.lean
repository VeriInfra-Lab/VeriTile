import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Unrolled

namespace VeriTile.Bench.Repairs.MultinomialSampling.UnrolledPair
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
variable {α : Type} [Inhabited α]

/-- A concrete single source invocation of this fixed complete UndefProgramIO.
The public memory and fallback inputs, domain and observations stay fixed. -/
structure Implementation (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) where
  invocation : Invocation α

def Equivalent {rest : Algebra α} {P : Primitives α} {p : Parameters} {pid : Nat}
    (left right : Implementation rest P p pid) : Prop :=
  let base:=io rest P p pid
  ∀ vals,Interface.SeedDomain rest P p pid vals → ∀ cap,base.InBounds cap vals →
    ∀ initial,base.InputBinding vals initial →
    left.invocation.Safe cap initial ∧ right.invocation.Safe cap initial ∧
    ∃ l r,left.invocation.run initial=some l ∧ right.invocation.run initial=some r ∧
      (∀ o j,base.outputMask o vals.memory j →
        mapCell (base.observe o j) (l.mem base.outputs[o].buf (base.write o vals.memory j))=
          mapCell (base.observe o j) (r.mem base.outputs[o].buf (base.write o vals.memory j))) ∧
      base.Frame vals initial l ∧ base.Frame vals initial r

instance (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) :
    VeriTile.Spec.ProgramSyntax (Implementation rest P p pid) where
  Statement:=Invocation α
  Signature:=Unit
  signature:=fun _=>()
  body:=fun prog=>[prog.invocation]
  structural:=some Equivalent
  sameContext:=Eq

def original (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) :
    Implementation rest P p pid := ⟨job rest P p pid⟩

def candidate (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) :
    Implementation rest P p pid := ⟨{job rest P p pid with kernel:=Unrolled.kernel p}⟩

end
end VeriTile.Bench.Repairs.MultinomialSampling.UnrolledPair

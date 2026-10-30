import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Native

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.NativeProfile
open VeriTile.Triton FP.Structural
open Native
noncomputable section
abbrev Operations := Ksoftmax.Graph.Operations
abbrev Expr := Ksoftmax.Graph.Expr
abbrev LocalExact := Ksoftmax.Words.LocalExact
abbrev NumericDomain := Ksoftmax.Words.NumericDomain

def lower (scalar : ℝ → Word) (S : Schedule) (consumer : Consumer) (t : Term) : Expr :=
  Scalar.lower scalar (S consumer) (t consumer)

theorem lower_exact (ops : Operations) (scalar : ℝ → Word) (S : Schedule)
    (consumer : Consumer) (t : Term) :
    Scalar.eval (Scalar.instructions ops scalar) (S consumer) (t consumer)=Ksoftmax.Graph.eval ops (lower scalar S consumer t) :=
  SoftmaxFlaggemsFiniteSeed.NativeProfile.lower_exact ops scalar (S consumer) (t consumer)

theorem observe_exact (ops : Operations) (scalar : ℝ → Word) (S : Schedule)
    (s : State Term) (r : RegionName) (o : Nat) :
    observe (Scalar.instructions ops scalar) S s r o=
      Ksoftmax.Graph.eval ops (lower scalar S (r,o) ((s.mem r o).read .real)) :=
  lower_exact ops scalar S (r,o) _

theorem exact_if_local (ops : Operations) (scalar : ℝ → Word) (S : Schedule)
    (consumer : Consumer) (ref : Ksoftmax.Words.PrimitiveReference) (ports : LocalExact ops ref) (t : Term)
    (domain : NumericDomain ops (lower scalar S consumer t)) :
    Scalar.eval (Scalar.instructions ops scalar) (S consumer) (t consumer)=
      Ksoftmax.Graph.eval (Ksoftmax.Words.referenceOps ref) (lower scalar S consumer t) :=
  (lower_exact ops scalar S consumer t).trans (Ksoftmax.Words.local_exact ops ref ports _ domain)

#axiomsClean lower_exact
#axiomsClean observe_exact
#axiomsClean exact_if_local
#specNonCircular lower avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.NativeProfile

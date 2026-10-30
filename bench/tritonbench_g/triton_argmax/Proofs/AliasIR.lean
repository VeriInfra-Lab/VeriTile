import bench.tritonbench_g.triton_argmax.Proofs.ScalarIR

/-! Width-aware related derivations. Each step follows a checked local native
transport or an explicit sign extension; no numerical equality is a premise. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Revision004.AliasIR
open SelectedEmission ScalarIR
variable {n : Nat} {graph : Graph n} {types : Ref → Kind}

inductive Value : {a b : Ref} → Expr valueBits graph types .value a → Expr valueBits graph types .value b → Prop where
  | refl (expression : Expr valueBits graph types .value a) : Value expression expression
  | transport (expression : Expr valueBits graph types .value a) (site : Ref) (typed : types site=.value)
      (edge : NativeTransport.Checked graph valueBits site a) : Value (.transport site expression typed edge) expression
  | symm {left : Expr valueBits graph types .value a} {right : Expr valueBits graph types .value b} : Value left right → Value right left
  | trans {left : Expr valueBits graph types .value a} {middle : Expr valueBits graph types .value b}
      {right : Expr valueBits graph types .value c} : Value left middle → Value middle right → Value left right

inductive Index : {wa wb : Nat} → {a b : Ref} →
    Expr valueBits graph types (.index wa) a → Expr valueBits graph types (.index wb) b → Prop where
  | refl (expression : Expr valueBits graph types (.index width) a) : Index expression expression
  | transport (expression : Expr valueBits graph types (.index width) a) (site : Ref) (typed : types site=.index width)
      (edge : NativeTransport.Checked graph width site a) : Index (.transport site expression typed edge) expression
  | widen (expression : Expr valueBits graph types (.index width) a) (site : Ref) (larger : width≤target)
      (typed : types site=.index target) (instruction : graph.instruction site=some (.copy a))
      (earlier : graph.time a<graph.time site) : Index (.widen site expression larger typed instruction earlier) expression
  | symm {left : Expr valueBits graph types (.index wa) a} {right : Expr valueBits graph types (.index wb) b} :
      Index left right → Index right left
  | trans {left : Expr valueBits graph types (.index wa) a} {middle : Expr valueBits graph types (.index wb) b}
      {right : Expr valueBits graph types (.index wc) c} : Index left middle → Index middle right → Index left right

theorem Value.evaluated (related : Value left right) (gt eq : α → α → Bool) (input : Fin n → α) :
    left.eval gt eq input=right.eval gt eq input := by
  induction related with
  | refl => rfl
  | transport => rfl
  | symm related ih => exact ih.symm
  | trans first second ihFirst ihSecond => exact ihFirst.trans ihSecond

theorem Index.evaluated (related : Index left right) (gt eq : α → α → Bool) (input : Fin n → α) :
    (left.eval gt eq input).toNat=(right.eval gt eq input).toNat := by
  induction related with
  | refl => rfl
  | transport => rfl
  | widen expression site larger typed instruction earlier =>
    have signed := BitVec.toInt_signExtend_of_le larger (x:=expression.eval gt eq input)
    rw [← show (Expr.widen site expression larger typed instruction earlier).eval gt eq input=
      (expression.eval gt eq input).signExtend _ from rfl] at signed
    rw [index_signed_natural,index_signed_natural] at signed
    exact Int.ofNat.inj signed
  | symm related ih => exact ih.symm
  | trans first second ihFirst ihSecond => exact ihFirst.trans ihSecond

theorem Value.source {left : Expr valueBits graph types .value a} {right : Expr valueBits graph types .value b}
    (related : Value left right) : Alias graph a b := by
  induction related with
  | refl => exact .refl _
  | transport expression site typed edge => exact .transport edge.source
  | symm related ih => exact .symm ih
  | trans first second ihFirst ihSecond => exact .trans ihFirst ihSecond

theorem Index.source {left : Expr valueBits graph types (.index wa) a} {right : Expr valueBits graph types (.index wb) b}
    (related : Index left right) : Alias graph a b := by
  induction related with
  | refl => exact .refl _
  | transport expression site typed edge => exact .transport edge.source
  | widen expression site larger typed instruction earlier => exact .transport (.copy instruction earlier)
  | symm related ih => exact .symm ih
  | trans first second ihFirst ihSecond => exact .trans ihFirst ihSecond

#axiomsClean Value.evaluated
#axiomsClean Index.evaluated
#axiomsClean Value.source
#axiomsClean Index.source
end VeriTile.Bench.Repairs.TritonArgmax.Revision004.AliasIR

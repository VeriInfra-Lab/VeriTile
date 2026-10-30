import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SelectedPlan

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
noncomputable section

/-- Additional obligations are individual scalar constant conversions. -/
structure ConstantPorts (n : Numerics) : Prop where
  naturalOne : n.natural 1=n.literal 1
  literalZero : n.literal 0=0

/-- Only the observed compiler's local rewrites. Congruence composes them;
there is deliberately no associative or whole-reduction constructor. -/
inductive LocalRewrite : ScalarGraph width → ScalarGraph width → Prop
  | refl (g) : LocalRewrite g g
  | symm {a b} : LocalRewrite a b → LocalRewrite b a
  | trans {a b c} : LocalRewrite a b → LocalRewrite b c → LocalRewrite a c
  | binary {a a' b b'} (op) : LocalRewrite a a' → LocalRewrite b b' →
      LocalRewrite (.binary op a b) (.binary op a' b')
  | unary {a a'} (op) : LocalRewrite a a' → LocalRewrite (.unary op a) (.unary op a')
  | fma {a a' b b' c c'} : LocalRewrite a a' → LocalRewrite b b' → LocalRewrite c c' →
      LocalRewrite (.fma a b c) (.fma a' b' c')
  | hostScale {a a'} (scale) : LocalRewrite a a' → LocalRewrite (.hostScale scale a) (.hostScale scale a')
  | transfer {a a'} (edge) : LocalRewrite a a' → LocalRewrite (.transfer edge a) (.transfer edge a')
  | addComm (a b) : LocalRewrite (.binary .add a b) (.binary .add b a)
  | mulComm (a b) : LocalRewrite (.binary .mul a b) (.binary .mul b a)
  | fmaComm (a b c) : LocalRewrite (.fma a b c) (.fma b a c)
  | twice (a) : LocalRewrite (.binary .mul (.literal 2) a) (.binary .add a a)
  | minusOne (a) : LocalRewrite (.binary .sub a (.literal 1)) (.binary .add a (.literal (-1)))
  | naturalOne : LocalRewrite (.natural 1) (.literal 1)
  | literalZero : LocalRewrite (.literal 0) (.raw 0)
  | move (edge a) : LocalRewrite (.transfer edge a) a

theorem LocalRewrite.value {n : Numerics} (laws : Peepholes n) (constants : ConstantPorts n)
    {a b : ScalarGraph width} (rewrite : LocalRewrite a b) (xs : Fin width → Word) :
    a.eval n xs=b.eval n xs := by
  induction rewrite with
  | refl => rfl
  | symm _ ih => exact ih.symm
  | trans _ _ ia ib => exact ia.trans ib
  | binary op _ _ ia ib => simp only [ScalarGraph.eval,ia,ib]
  | unary op _ ih => simp only [ScalarGraph.eval,ih]
  | fma _ _ _ ia ib ic => simp only [ScalarGraph.eval,ia,ib,ic]
  | hostScale scale _ ih => simp only [ScalarGraph.eval,ih]
  | transfer edge _ ih => exact ih
  | addComm a b => exact laws.addComm _ _
  | mulComm a b => exact laws.mulComm _ _
  | fmaComm a b c => exact laws.fmaComm _ _ _
  | twice a => exact laws.twice _
  | minusOne a => exact laws.minusOne _
  | naturalOne => exact constants.naturalOne
  | literalZero => exact constants.literalZero
  | move => rfl

/-- The actual object graph may differ by verified local instruction changes.
The source expression, selected reductions and their input origins remain
explicit in `plan`; neither graph contains a whole-reduction numeric port. -/
structure CompiledExecution {σ : Type} {platform : Selection σ} {expression : Expr}
    (plan : SelectedPlan platform expression) (trace : PrimitiveTrace σ)
    (readWord : σ → VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Tensor → Nat → Word) where
  nativeGraph : LocatedGraph σ 0
  lowering : LocalRewrite nativeGraph.erase plan.compile.value.erase
  origins : ∀ site ∈ plan.compile.value.sites,
    ∃ actual ∈ nativeGraph.sites, actual.call=site.call ∧ actual.coordinate=site.coordinate
  output : Nat
  execution : LocatedExecution trace readWord Fin.elim0 nativeGraph output

theorem CompiledExecution.reference_value {σ : Type} {platform : Selection σ} {expression : Expr}
    {plan : SelectedPlan platform expression} {trace : PrimitiveTrace σ} {native reference : Numerics}
    {readWord readByte} (api : PrimitiveContracts trace native readWord readByte)
    (ports : Obligations native reference) (laws : Peepholes native) (constants : ConstantPorts native)
    (run : CompiledExecution plan trace readWord) : trace.bits run.output=plan.reference reference := by
  exact (run.execution.erase.native_value api).trans
    ((run.lowering.value laws constants Fin.elim0).trans
      (plan.compile.value.erase.equiv ports Fin.elim0 Fin.elim0 (fun i => Fin.elim0 i)))

#axiomsClean LocalRewrite.value
#axiomsClean CompiledExecution.reference_value
#specNonCircular LocalRewrite avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating

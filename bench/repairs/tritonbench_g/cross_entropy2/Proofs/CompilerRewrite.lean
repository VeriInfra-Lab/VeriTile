import bench.repairs.tritonbench_g.cross_entropy2.Proofs.CompilerSites

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source
noncomputable section
set_option maxHeartbeats 6000000

/-- A proof-producing local compiler certificate. Constructors are either
structural congruences or a single explicitly named scalar rewrite. There
is no constructor admitting an arbitrary output/expression equality. -/
inductive Rewrite : Instruction → Instruction → Prop
  | refl (x) : Rewrite x x
  | trans {a b c} : Rewrite a b → Rewrite b c → Rewrite a c
  | binary (op) {a b x y} : Rewrite a x → Rewrite b y → Rewrite (.binary op a b) (.binary op x y)
  | unary (op) {a b} : Rewrite a b → Rewrite (.unary op a) (.unary op b)
  | fma {a b c x y z} : Rewrite a x → Rewrite b y → Rewrite c z → Rewrite (.fma a b c) (.fma x y z)
  | negation {a b} : Rewrite a b → Rewrite (.negation a) (.negation b)
  | hostAdd {a b x y} : Rewrite a x → Rewrite b y → Rewrite (.hostAdd a b) (.hostAdd x y)
  | hostMultiply {a b x y} : Rewrite a x → Rewrite b y → Rewrite (.hostMultiply a b) (.hostMultiply x y)
  | hostSquare {a b} : Rewrite a b → Rewrite (.hostSquare a) (.hostSquare b)
  | hostScale (scale) {a b} : Rewrite a b → Rewrite (.hostScale scale a) (.hostScale scale b)
  | sum (width) {a b : Fin width → Instruction} : (∀ i, Rewrite (a i) (b i)) → Rewrite (.sum width a) (.sum width b)
  | maximum (width) {a b : Fin width → Instruction} : (∀ i, Rewrite (a i) (b i)) → Rewrite (.maximum width a) (.maximum width b)
  | hostSum (width) {a b : Fin width → Instruction} : (∀ i, Rewrite (a i) (b i)) → Rewrite (.hostSum width a) (.hostSum width b)
  | hostLSE (width) {a b : Fin width → Instruction} : (∀ i, Rewrite (a i) (b i)) → Rewrite (.hostLSE width a) (.hostLSE width b)
  | rankSum (width) {a b : Fin width → Instruction} : (∀ i, Rewrite (a i) (b i)) → Rewrite (.rankSum width a) (.rankSum width b)
  | addSwap (a b) : Rewrite (.binary .add a b) (.binary .add b a)
  | mulSwap (a b) : Rewrite (.binary .mul a b) (.binary .mul b a)
  | fmaSwap (a b c) : Rewrite (.fma a b c) (.fma b a c)
  | negFmaSwap (a b c) : Rewrite (.fma (.negation a) b c) (.fma (.negation b) a c)
  | twice (x) : Rewrite (.binary .mul (.literal 2) x) (.binary .add x x)
  | minusOne (x) : Rewrite (.binary .sub x (.literal 1)) (.binary .add x (.literal (-1)))
  | naturalOne : Rewrite (.natural 1) (.literal 1)

/-- Every complete compiler certificate reduces to local scalar numerical
laws. Reduction plans and seeds are preserved structurally through it. -/
theorem Rewrite.sound (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    {source native : Instruction} (certificate : Rewrite source native) :
    runInstruction n profile source=runInstruction n profile native := by
  induction certificate with
  | refl => rfl
  | trans ab bc ia ib => exact ia.trans ib
  | binary op a b ia ib => cases op <;> simp only [runInstruction,ia,ib]
  | unary op a ia => cases op <;> simp only [runInstruction,ia]
  | fma a b c ia ib ic => simp only [runInstruction,ia,ib,ic]
  | negation a ia | hostSquare a ia => simp only [runInstruction,ia]
  | hostAdd a b ia ib | hostMultiply a b ia ib => simp only [runInstruction,ia,ib]
  | hostScale scale a ia => simp only [runInstruction,ia]
  | sum width steps ih | maximum width steps ih | hostSum width steps ih | hostLSE width steps ih | rankSum width steps ih =>
    simp only [runInstruction,funext ih]
  | addSwap a b => exact laws.addComm _ _
  | mulSwap a b => exact laws.mulComm _ _
  | fmaSwap a b c => exact laws.fmaComm _ _ _
  | negFmaSwap a b c => exact laws.negFmaExchange _ _ _
  | twice x => exact laws.twice _
  | minusOne x => exact laws.minusOne _
  | naturalOne => exact one

theorem certified_expression (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (source : Expr) (native : Instruction) (certificate : Rewrite (compile source).value native) :
    word n profile source=runInstruction n profile native :=
  (compiled_word n profile source).symm.trans (certificate.sound n profile laws one)

#axiomsClean Rewrite.sound
#axiomsClean certified_expression
#specNonCircular Rewrite avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating

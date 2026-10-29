import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScalarPrograms

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source
noncomputable section
set_option maxHeartbeats 10000000

def nativeScaled (n : Ports) (p : Parameters) (block : Nat) (input : Fin p.block → Word) (lane : Fin p.block) : Word :=
  if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then
    n.mul (n.literal p.logitScale) (input lane) else n.negativeInfinity

def nativeMaximum (n : Ports) (profile : Profile) {width : Nat} (xs : Fin width → Word) : Word :=
  reduce profile.maxTree n.max n.negativeInfinity (List.ofFn xs)
def nativeLSE (n : Ports) (profile : Profile) {width : Nat} (xs : Fin width → Word) : Word :=
  let maximum := nativeMaximum n profile xs
  n.add maximum (n.log (reduce profile.sumTree n.add (n.literal 0)
    (List.ofFn (fun lane => n.exp (n.sub (xs lane) maximum)))))

def lseExpression {width : Nat} (xs : Fin width → Expr) : Expr :=
  .binary .add (.unary .log (.sum width (fun lane =>
    .unary .exp (.binary .sub (xs lane) (.maximum width xs))))) (.maximum width xs)

theorem lse_expression_native (n : Ports) (profile : Profile) (laws : Peepholes n)
    {width : Nat} (xs : Fin width → Expr) :
    word n profile (lseExpression xs)=nativeLSE n profile (fun i => word n profile (xs i)) := by
  change n.add (n.log (reduce profile.sumTree n.add (n.literal 0)
    (List.ofFn (fun i => n.exp (n.sub (word n profile (xs i))
      (nativeMaximum n profile (fun j => word n profile (xs j))))))))
    (nativeMaximum n profile (fun i => word n profile (xs i)))=_
  exact laws.addComm _ _

/-- Complete forward reduction graph. Every active scaled input, maximum,
exponential leaf, sum and logarithm is retained; the scalar source ordering
is transported to the pinned native operand choices by local laws. -/
theorem forward_reductions_native (n : Ports) (profile : Profile) (laws : Peepholes n)
    (p : Parameters) (memory : Memory Expr) (row block : Nat) :
    let input := fun lane : Fin p.block => word n profile
      ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real)
    let xs := nativeScaled n p block input
    word n profile (values algebra p memory row block).lse=nativeLSE n profile xs ∧
    word n profile ((values algebra p memory row block).sumLogits.getD (algebra.literal 0))=
      if p.hasSmoothing then reduce profile.sumTree n.add (n.literal 0) (List.ofFn (fun lane =>
        if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then xs lane else n.literal 0))
      else n.literal 0 := by
  dsimp only
  let sourceXs := VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits (scaled algebra p.logitScale) p.toParameters memory row block
  let input := fun lane : Fin p.block => word n profile
    ((memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row
      (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))).read .real)
  let xs := nativeScaled n p block input
  have lanes : ∀ lane, word n profile (sourceXs lane)=xs lane := by
    intro lane
    dsimp only [sourceXs,VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits,nativeScaled,xs]
    split
    · exact laws.mulComm _ _
    · rfl
  constructor
  · have lseEq : (values algebra p memory row block).lse=lseExpression sourceXs := rfl
    rw [lseEq,lse_expression_native n profile laws]
    exact congrArg (nativeLSE n profile) (funext lanes)
  · have maskLanes : ∀ lane, word n profile
        (if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then sourceXs lane else .literal 0)=
        if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then xs lane else n.literal 0 := by
      intro lane
      split
      · exact lanes lane
      · rfl
    cases smoothing : p.hasSmoothing with
    | «false» =>
      simp only [values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,
        smoothing,Bool.false_eq_true,↓reduceIte,Option.getD]
      rfl
    | «true» =>
      simp only [values,VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues,
        smoothing,↓reduceIte,Option.getD]
      change reduce profile.sumTree n.add (n.literal 0) (List.ofFn (fun lane => word n profile
        (if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then sourceXs lane else .literal 0)))=_
      exact congrArg (fun f : Fin p.block → Word => reduce profile.sumTree n.add (n.literal 0) (List.ofFn f)) (funext maskLanes)

#axiomsClean lse_expression_native
#axiomsClean forward_reductions_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating

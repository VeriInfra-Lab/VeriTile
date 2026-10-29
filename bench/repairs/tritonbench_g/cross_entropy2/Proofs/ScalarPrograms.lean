import bench.repairs.tritonbench_g.cross_entropy2.Proofs.CompilerRewrite
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.WholeGraph

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source
noncomputable section
set_option maxHeartbeats 10000000

/-- Native forward scalar instructions after the complete reductions. The
operand orders are those of the pinned SM90 PTX, including the separately
stored z product and the FMA used only for the corrected loss. -/
def nativeLossZ (n : Ports) (p : Parameters) (target : Int) (owner : Prop) [Decidable owner]
    (L sums selected : Word) : Word × Word :=
  if target=p.ignored then (n.literal 0,n.literal 0) else
  let base := if p.split then n.literal 0 else L
  let smooth := n.literal p.smoothing
  let loss := if owner then
    if p.hasSmoothing then
      n.fma (n.neg (n.sub (n.literal 1) smooth)) (n.mul (n.literal p.logitScale) selected)
        (n.sub base (n.div (n.mul smooth sums) (n.natural p.totalClasses)))
    else n.fma (n.neg (n.literal p.logitScale)) selected base
  else if p.hasSmoothing then
    let original := n.mul smooth (n.sub base (n.div sums (n.natural p.totalClasses)))
    if p.split then original else n.fma (n.sub (n.literal 1) smooth) L original
  else base
  if p.split then (loss,n.literal 0) else
    (n.fma L (n.mul (n.literal p.squareScale) L) loss,n.mul L (n.mul (n.literal p.squareScale) L))

/-- All complete forward scalar branches follow from the actual source
loss/z expressions. No whole-output or successful-kernel equality is a
hypothesis; the only numerical premises are individual scalar laws. -/
theorem loss_z_native (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (p : Parameters) (memory : Memory Expr) (row block : Nat) (target : Int)
    (v : VeriTile.Bench.Repairs.CrossEntropy1.Source.ForwardValues Expr p.block) :
    (word n profile (lossZ algebra p memory row block target v).1,
      word n profile (lossZ algebra p memory row block target v).2)=
    nativeLossZ n p target (blockOwns p target block) (word n profile v.lse)
      (word n profile (v.sumLogits.getD (algebra.literal 0)))
      (word n profile ((memory p.logits
        (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat)).read .real)) := by
  change n.natural 1=n.literal 1 at one
  by_cases ignored : target=p.ignored
  · simp [lossZ,nativeLossZ,ignored,word,fp,VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,algebra,evaluate]
  · cases split : p.split <;> cases smoothing : p.hasSmoothing <;>
      by_cases owner : blockOwns p target block
    all_goals simp only [lossZ,nativeLossZ,ignored,owner,split,smoothing,↓reduceIte,
      Bool.false_eq_true,VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,
      VeriTile.Bench.Repairs.CrossEntropy1.Source.nat,VeriTile.Bench.Repairs.CrossEntropy1.Source.load,
      VeriTile.Bench.Repairs.CrossEntropy1.Source.binary,algebra,word,evaluate,add,sub]
    all_goals try rw [laws.negFmaExchange _ (n.literal p.logitScale)]
    all_goals simp only [one,laws.fmaComm (n.mul _ _) _,laws.mulComm (n.mul _ _) _,laws.mulComm _ (n.literal p.logitScale)]

#axiomsClean loss_z_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating

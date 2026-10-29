import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Memory
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Transport

namespace VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section
set_option maxHeartbeats 6000000

inductive InputKind where
  | single | rms (byte : Bool) | weight (byte : Bool)
def inputEval (a : StorageOps F H) (kind : InputKind) (m : Memory F H) (p : Address) : F :=
  match kind with
  | .single => read32 a m p
  | .rms byte => loadRms a byte m p
  | .weight byte => loadWeight a byte m p

/-- Scalar SSA expression. The only arity>3 leaf is the fixed K8 instruction;
all source K loops, reduction nodes and rotation operations are explicit. -/
inductive Expr (F : Type) where
  | literal (v : F)
  | input (kind : InputKind) (p : Address)
  | zero | one
  | add (x y : Expr F)
  | mul (x y : Expr F)
  | mulFTZ (x y : Expr F)
  | divide (x y : Expr F)
  | rsqrt (x : Expr F)
  | neg (x : Expr F)
  | fma (x y z : Expr F)
  | ofInt (n : Int)
  | pow (x y : Expr F)
  | cos (x : Expr F)
  | sin (x : Expr F)
  | mma8 (x y : Fin 8 → Expr F) (old : Expr F)

def exprOps : Ops (Expr F) where
  zero := .zero
  one := .one
  add := .add
  mul := .mul
  mulFTZ := .mulFTZ
  divide := .divide
  rsqrt := .rsqrt
  neg := .neg
  fma := .fma
  ofInt := .ofInt
  pow := .pow
  cos := .cos
  sin := .sin
  mma8 := .mma8
def Expr.eval (a : StorageOps F H) (m : Memory F H) : Expr F → F
  | .literal v => v
  | .input kind p => inputEval a kind m p
  | .zero => a.zero
  | .one => a.one
  | .add x y => a.add (x.eval a m) (y.eval a m)
  | .mul x y => a.mul (x.eval a m) (y.eval a m)
  | .mulFTZ x y => a.mulFTZ (x.eval a m) (y.eval a m)
  | .divide x y => a.divide (x.eval a m) (y.eval a m)
  | .rsqrt x => a.rsqrt (x.eval a m)
  | .neg x => a.neg (x.eval a m)
  | .fma x y z => a.fma (x.eval a m) (y.eval a m) (z.eval a m)
  | .ofInt n => a.ofInt n
  | .pow x y => a.pow (x.eval a m) (y.eval a m)
  | .cos x => a.cos (x.eval a m)
  | .sin x => a.sin (x.eval a m)
  | .mma8 x y old => a.mma8 (fun i => (x i).eval a m) (fun i => (y i).eval a m) (old.eval a m)
def exprHom (a : StorageOps F H) (m : Memory F H) :
    exprOps.Hom a.toOps (Expr.eval a m) where
  zero := rfl
  one := rfl
  add := fun _ _ => rfl
  mul := fun _ _ => rfl
  mulFTZ := fun _ _ => rfl
  divide := fun _ _ => rfl
  rsqrt := fun _ => rfl
  neg := fun _ => rfl
  fma := fun _ _ _ => rfl
  ofInt := fun _ => rfl
  pow := fun _ _ => rfl
  cos := fun _ => rfl
  sin := fun _ => rfl
  mma8 := fun _ _ _ => rfl

def inputExpr (b : Buffers) (phase : Fin 3) (batch row : Nat) : RowInput (Expr F) where
  x := fun k => .input .single (b.x.address batch row k)
  rms := fun k => .input (.rms b.useByte) (b.rms.address k)
  weight := fun n k => .input (.weight b.useByte) ((b.weight phase).address n k)
def cellExpr (p : Parameters F) (b : Buffers)
    (plans : Fin 3 → Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Fin 3 → Nat → Nat → Reduction) (x : Cell s) : Expr F :=
  value exprOps (p.map Expr.literal) (inputExpr b x.1 x.2.1 x.2.2.1)
    (plans x.1 x.2.1 x.2.2.1) (reductions x.1 x.2.1 x.2.2.1)
    (rotary x.1) x.2.2.1 x.2.2.2
theorem cellExpr_eval (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Fin 3 → Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Fin 3 → Nat → Nat → Reduction) (m : Memory F H) (x : Cell s) :
    encode a b x.1 ((cellExpr p b plans reductions x).eval a m)=cellValue a p b plans reductions m x := by
  unfold cellExpr
  rw [value_map exprOps a.toOps (Expr.eval a m) (exprHom a m)]
  rfl

#axiomsClean cellExpr_eval
end
end VeriTile.Bench.Repairs.RmsMatmulRbe

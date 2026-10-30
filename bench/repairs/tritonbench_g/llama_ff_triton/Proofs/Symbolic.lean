import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Memory
import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Transport

namespace VeriTile.Bench.Repairs.LlamaFF
open VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 6000000

inductive InputKind where
  | single | rms (byte : Bool) | weight (byte : Bool)
def inputEval (a : StorageOps F H) (kind : InputKind) (m : Memory F H) (p : Address) : F :=
  match kind with
  | .single => a.widen (read16 a m p)
  | .rms byte => loadRms a byte m p
  | .weight byte => loadWeight a byte m p

/-- Scalar SSA expression. The only arity>3 leaf is the fixed K16 instruction;
all source K loops, reduction nodes and rotation operations are explicit. -/
inductive Expr (F : Type) where
  | literal (v : F)
  | input (kind : InputKind) (p : Address)
  | zero | one
  | add (x y : Expr F)
  | mulInput (x y : Expr F)
  | sigmoidNeg (x : Expr F)
  | neg (x : Expr F)
  | fma (x y z : Expr F)
  | mul (x y : Expr F)
  | mulFTZ (x y : Expr F)
  | divide (x y : Expr F)
  | rsqrt (x : Expr F)
  | ofInt (n : Int)
  | mma16 (x y : Fin 16 → Expr F) (old : Expr F)

def exprOps : Ops (Expr F) where
  zero := .zero
  one := .one
  add := .add
  mulInput := .mulInput
  sigmoidNeg := .sigmoidNeg
  neg := .neg
  fma := .fma
  mul := .mul
  mulFTZ := .mulFTZ
  divide := .divide
  rsqrt := .rsqrt
  ofInt := .ofInt
  mma16 := .mma16
def Expr.eval (a : StorageOps F H) (m : Memory F H) : Expr F → F
  | .literal v => v
  | .input kind p => inputEval a kind m p
  | .zero => a.zero
  | .one => a.one
  | .add x y => a.add (x.eval a m) (y.eval a m)
  | .mulInput x y => a.mulInput (x.eval a m) (y.eval a m)
  | .sigmoidNeg x => a.sigmoidNeg (x.eval a m)
  | .neg x => a.neg (x.eval a m)
  | .fma x y z => a.fma (x.eval a m) (y.eval a m) (z.eval a m)
  | .mul x y => a.mul (x.eval a m) (y.eval a m)
  | .mulFTZ x y => a.mulFTZ (x.eval a m) (y.eval a m)
  | .divide x y => a.divide (x.eval a m) (y.eval a m)
  | .rsqrt x => a.rsqrt (x.eval a m)


  | .ofInt n => a.ofInt n



  | .mma16 x y old => a.mma16 (fun i => (x i).eval a m) (fun i => (y i).eval a m) (old.eval a m)
def exprHom (a : StorageOps F H) (m : Memory F H) :
    exprOps.Hom a.toOps (Expr.eval a m) where
  zero := rfl
  one := rfl
  add := fun _ _ => rfl
  mulInput := fun _ _ => rfl
  sigmoidNeg := fun _ => rfl
  neg := fun _ => rfl
  fma := fun _ _ _ => rfl
  mul := fun _ _ => rfl
  mulFTZ := fun _ _ => rfl
  divide := fun _ _ => rfl
  rsqrt := fun _ => rfl
  ofInt := fun _ => rfl
  mma16 := fun _ _ _ => rfl

def inputExpr (b : Buffers) (batch row : Nat) : RowInput (Expr F) where
  x := fun k => .input .single (b.x.address batch row k)
  rms := fun k => .input (.rms b.useByte) (b.rms.address k)
  weight := fun n k => .input (.weight b.useByte) (b.weight.address n k)
  weight3 := fun n k => .input (.weight b.useByte) (b.weight3.address n k)
def cellExpr (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (x : Cell s) : Expr F :=
  value exprOps (p.map Expr.literal) (inputExpr b x.1 x.2.1)
    (plans x.1 x.2.1) (reductions x.1 x.2.1)
    x.2.2
theorem cellExpr_eval (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (m : Memory F H) (x : Cell s) :
    encode a b ((cellExpr p b plans reductions x).eval a m)=cellValue a p b plans reductions m x := by
  unfold cellExpr
  rw [value_map exprOps a.toOps (Expr.eval a m) (exprHom a m)]
  rfl

#axiomsClean cellExpr_eval
end
end VeriTile.Bench.Repairs.LlamaFF

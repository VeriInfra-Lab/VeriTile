import bench.tritonbench_g.layer_norm_triton.Proofs.Numerics

namespace VeriTile.Bench.TritonBenchG.LayerNormLock
open VeriTile.Triton
open FP.Equational (ReductionPlan)

inductive Read where | x | dy | w | mean | rstd
  deriving DecidableEq, Repr
inductive Expr where
  | reg (name : Nat)
  | read (kind : Read)
  | zero
  | denominator
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | div (a b : Expr)
  | cast (src dst : Storage) (a : Expr)
  | mask (a : Expr)
  | sum (a : Expr)
  deriving DecidableEq, Repr

abbrev Registers (V : Type) := Nat → Nat → V

def rowRead (b : Backend V) (c : Config) (mem : Memory V) (undef : Undef V)
    (p : Nat) (kind : Read) (lane : Nat) : V :=
  match kind with
  | .x => input32 b c mem .x c.dtype.x p 0 lane
  | .dy => input32 b c mem .dy c.dtype.dy p 0 lane
  | .w => b.cast c.dtype.w .fp32
      (if live c 0 lane then mem .w (BitVec.ofNat 32 lane).toInt else undef .w (BitVec.ofNat 32 lane).toInt)
  | .mean => mem .mean (BitVec.ofNat 32 p).toInt
  | .rstd => mem .rstd (BitVec.ofNat 32 p).toInt

def eval (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (p : Nat) (env : Registers V) : Expr → Nat → V
  | .reg name, lane => env name lane
  | .read kind, lane => rowRead b c mem undef p kind lane
  | .zero, _ => b.zero
  | .denominator, _ => b.fromI32 c.n
  | .add a d, lane => b.add (eval b c plan mem undef p env a lane) (eval b c plan mem undef p env d lane)
  | .sub a d, lane => b.sub (eval b c plan mem undef p env a lane) (eval b c plan mem undef p env d lane)
  | .mul a d, lane => b.mul (eval b c plan mem undef p env a lane) (eval b c plan mem undef p env d lane)
  | .div a d, lane => b.div (eval b c plan mem undef p env a lane) (eval b c plan mem undef p env d lane)
  | .cast src dst a, lane => b.cast src dst (eval b c plan mem undef p env a lane)
  | .mask a, lane => if live c 0 lane then eval b c plan mem undef p env a lane else b.zero
  | .sum a, _ => sum b plan (eval b c plan mem undef p env a)

abbrev Assignment := Nat × Expr

/-- Original assignment order, including both overwritten xhat/wdy registers,
all padded arithmetic before their masks, and explicit same-fp32 .to(w.dtype).
DX storage conversion occurs separately at the actual output store. -/
def backwardProgram : List Assignment := [
  (0, .read .x), (1, .read .dy), (2, .read .w), (3, .read .mean), (4, .read .rstd),
  (5, .mul (.sub (.reg 0) (.reg 3)) (.reg 4)),
  (6, .mul (.reg 2) (.reg 1)),
  (5, .mask (.reg 5)), (6, .mask (.reg 6)),
  (7, .div (.sum (.mul (.reg 5) (.reg 6))) .denominator),
  (8, .div (.sum (.reg 6)) .denominator),
  (9, .mul (.sub (.reg 6) (.add (.mul (.reg 5) (.reg 7)) (.reg 8))) (.reg 4)),
  (10, .cast .fp32 .fp32 (.mul (.reg 1) (.reg 5))),
  (11, .cast .fp32 .fp32 (.reg 1))]

def runAssignments (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (p : Nat) : List Assignment → Registers V → Registers V
  | [], env => env
  | (name, rhs) :: rest, env =>
    runAssignments b c plan mem undef p rest (put env name (eval b c plan mem undef p env rhs))

structure RowValues (V : Type) where
  dx : Nat → V
  dw : Nat → V
  db : Nat → V

def backwardSource (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (p : Nat) : RowValues V :=
  let env := runAssignments b c plan mem undef p backwardProgram (fun _ _ => b.zero)
  ⟨fun lane => b.cast .fp32 c.dtype.dx (env 9 lane), env 10, env 11⟩

/-- Independent row formula; no source register execution appears here. -/
def backwardReference (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (p : Nat) : RowValues V :=
  let x := rowRead b c mem undef p .x
  let dy := rowRead b c mem undef p .dy
  let w := rowRead b c mem undef p .w
  let mean := mem .mean (BitVec.ofNat 32 p).toInt
  let rstd := mem .rstd (BitVec.ofNat 32 p).toInt
  let h := fun lane => if live c 0 lane then b.mul (b.sub (x lane) mean) rstd else b.zero
  let wd := fun lane => if live c 0 lane then b.mul (w lane) (dy lane) else b.zero
  let c1 := b.div (sum b plan (fun lane => b.mul (h lane) (wd lane))) (b.fromI32 c.n)
  let c2 := b.div (sum b plan wd) (b.fromI32 c.n)
  ⟨fun lane => b.cast .fp32 c.dtype.dx (b.mul (b.sub (wd lane) (b.add (b.mul (h lane) c1) c2)) rstd),
   fun lane => b.cast .fp32 .fp32 (b.mul (dy lane) (h lane)),
   fun lane => b.cast .fp32 .fp32 (dy lane)⟩

theorem backward_value (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (p : Nat) :
    backwardSource b c plan mem undef p = backwardReference b c plan mem undef p := by
  simp only [backwardSource, backwardProgram, runAssignments, eval, put]
  rfl

#axiomsClean backward_value
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock

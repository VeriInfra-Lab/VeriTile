import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPLogical

namespace VeriTile.Bench.Repairs.DecayCumsum.Native
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32
def word (n : Nat) : Word := BitVec.ofNat 32 n
def positiveZero : Word := word 0
def invLn2 : Word := word 0x3fb8aa3b

/-- Only individual PTX instruction interpretations are opaque. These ports
have no arithmetic identities and range over every raw FP32 word. -/
structure Instructions where
  addRN : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
  exp2ApproxFTZ : Word → Word
  negF32 : Word → Word

/-- Explicit host sign-bit codec. Native neg.f32 has its own instruction port
so its behavior on all NaN payloads is not assumed to be a bit operation. -/
def negBits (x : Word) : Word := x ^^^ word 0x80000000

inductive Expr where
  | arg (n : Nat) | const (w : Word)
  | add (a b : Expr) | sub (a b : Expr) | mul (a b : Expr)
  | fma (a b c : Expr) | neg (a : Expr) | exp2 (a : Expr)
  deriving DecidableEq

def Expr.eval (p : Instructions) (args : Nat → Word) : Expr → Word
  | .arg n => args n
  | .const w => w
  | .add a b => p.addRN (eval p args a) (eval p args b)
  | .sub a b => p.subRN (eval p args a) (eval p args b)
  | .mul a b => p.mulRN (eval p args a) (eval p args b)
  | .fma a b c => p.fmaRN (eval p args a) (eval p args b) (eval p args c)
  | .neg a => p.negF32 (eval p args a)
  | .exp2 a => p.exp2ApproxFTZ (eval p args a)

inductive Site where
  | forwardStep | prepareScale | prepareQ | prepareDifference | prepareK
  | backwardQ | backwardK | backwardDifference | gradientRight | gradient
  | firstCarry | laterCarry | exponentQ | exponentK
  deriving DecidableEq
abbrev Graph := Site → Expr

/-- Actual operand order, including the compiler's first reverse-carry
specialization. The first forward step remains an FMA with +0. -/
def selected : Graph
  | .forwardStep => .fma (.arg 0) (.const invLn2) (.arg 1)
  | .prepareScale => .mul (.arg 1) (.arg 0)
  | .prepareQ | .prepareK => .mul (.arg 1) (.arg 0)
  | .prepareDifference | .backwardDifference => .sub (.arg 0) (.arg 1)
  | .backwardQ | .backwardK => .fma (.arg 2) (.arg 1) (.arg 0)
  | .gradientRight => .mul (.arg 0) (.arg 1)
  | .gradient => .fma (.arg 0) (.arg 1) (.neg (.arg 2))
  | .firstCarry => .add (.arg 1) (.arg 0)
  | .laterCarry => .add (.arg 0) (.arg 1)
  | .exponentQ | .exponentK => .exp2 (.arg 0)

def arguments (a b c : Word) : Nat → Word
  | 0 => a | 1 => b | _ => c
def evalSite (p : Instructions) (graph : Graph) (site : Site) (a b : Word)
    (c : Word := positiveZero) : Word := (graph site).eval p (arguments a b c)

/-- Independent native recurrences. No AST, carrier, interpreter, source
execution, host result or output equality occurs in this reference. -/
def forwardWords (p : Instructions) (g : Nat → Word) : Nat → Word
  | 0 => positiveZero
  | n+1 => p.fmaRN (g n) invLn2 (forwardWords p g n)
def prepareQWord (p : Instructions) (scale q g : Word) : Word :=
  p.mulRN (p.mulRN scale (p.exp2ApproxFTZ g)) q
def prepareKWord (p : Instructions) (k last g : Word) : Word :=
  p.mulRN (p.exp2ApproxFTZ (p.subRN last g)) k
def backwardQWord (p : Instructions) (inner inter g : Word) : Word :=
  p.fmaRN (p.exp2ApproxFTZ g) inter inner
def backwardKWord (p : Instructions) (inner inter last g : Word) : Word :=
  p.fmaRN (p.exp2ApproxFTZ (p.subRN last g)) inter inner
def gradientWord (p : Instructions) (dq dk q k : Word) : Word :=
  p.fmaRN dq q (p.negF32 (p.mulRN dk k))
def backwardWords (p : Instructions) (span : Nat)
    (dqi dqo dki dko q k g : Nat → Word) : Nat → Word
  | 0 => positiveZero
  | n+1 =>
    let t := span-1-n
    let gradient := gradientWord p (backwardQWord p (dqi t) (dqo t) (g t))
      (backwardKWord p (dki t) (dko t) (g (span-1)) (g t)) (q t) (k t)
    if n=0 then p.addRN gradient positiveZero
    else p.addRN (backwardWords p span dqi dqo dki dko q k g n) gradient
def evaluateWords (p : Instructions) (kind : Kind) (span : Nat) (scale : Word)
    (read : RegionName → Nat → Word) (r : RegionName) (t : Nat) : Word :=
  match kind with
  | .forward => forwardWords p (read "g") (t+1)
  | .prepare => if r="qg" then prepareQWord p scale (read "q" t) (read "g" t)
      else prepareKWord p (read "k" t) (read "g" (span-1)) (read "g" t)
  | .backward => if r="dqo" then backwardQWord p (read "dqi" t) (read "dqo" t) (read "g" t)
      else if r="dko" then backwardKWord p (read "dki" t) (read "dko" t) (read "g" (span-1)) (read "g" t)
      else backwardWords p span (read "dqi") (read "dqo") (read "dki") (read "dko")
        (read "q") (read "k") (read "g") (span-t)
def referenceWords (p : Instructions) (h : Call) (kind : Kind) (scale : Word)
    (data : LogicalStorage.Data Word) (r : RegionName) (c : Logical.Coordinate h) : Word :=
  evaluateWords p kind (Logical.chunkSpan h c.2.1.val) scale
    (fun s n => data s c.1.val (Logical.chunkStart h c.2.1.val+n) c.2.2.val)
    r (c.2.1.val-Logical.chunkStart h c.2.1.val)

#specNonCircular referenceWords avoiding [FPLogical.reference, FPStorage.execute]
end
end VeriTile.Bench.Repairs.DecayCumsum.Native

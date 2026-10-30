import bench.tritonbench_g.cross_entropy1.Proofs.ReductionSchedules

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open Source
noncomputable section

/-- Source arithmetic graph. A barrier denotes a source load, so contraction
cannot cross tensor storage. Reductions retain their complete ordered lanes. -/
inductive Expr where
  | raw (bits : Word)
  | literal (value : ℝ)
  | natural (value : Nat)
  | negativeInfinity
  | barrier (value : Expr)
  | binary (operation : Binary32) (left right : Expr)
  | unary (operation : Unary32) (value : Expr)
  | sum (width : Nat) (values : Fin width → Expr)
  | maximum (width : Nat) (values : Fin width → Expr)
  | negation (value : Expr)
  | hostAdd (left right : Expr)
  | hostSquare (value : Expr)
  | hostScale (scale : ℝ) (value : Expr)
  | hostSum (width : Nat) (values : Fin width → Expr)
  | hostLSE (width : Nat) (values : Fin width → Expr)
  | rankSum (width : Nat) (values : Fin width → Expr)

instance : Inhabited Expr := ⟨.raw 0⟩

def algebra : FP32 Expr where
  literal := .literal
  fromNat := .natural
  negInf := .negativeInfinity
  load := .barrier
  add := .binary .add
  sub := .binary .sub
  mul := .binary .mul
  div := .binary .div
  exp := .unary .exp
  log := .unary .log
  sum := fun {width} values => .sum width values
  maximum := fun {width} values => .maximum width values

/-- The actual host operations each read stored operands. These nodes keep
Torch reduction and collective order separate from Triton lane reductions. -/
def hostArithmetic : Host.TensorArithmetic Expr where
  zero := .literal 0
  add := fun a b => .hostAdd (.barrier a) (.barrier b)
  square := fun x => .hostSquare (.barrier x)
  scale := fun scale x => .hostScale scale (.barrier x)
  sum := fun xs => .hostSum xs.length (fun i => .barrier (xs.get i))
  logsumexp := fun xs => .hostLSE xs.length (fun i => .barrier (xs.get i))
  rankSum := fun xs => .rankSum xs.length (fun i => .barrier (xs.get i))

/-- Retain the operands of the last multiply until its use is known. The
rounded product is available to ordinary consumers; a contracted add/sub uses
the original two operands. All other nodes, including loads and reductions,
end this local contraction opportunity. -/
structure Evaluated where
  value : Word
  product : Option (Word × Word) := none

def add (n : Numerics) (a b : Evaluated) : Evaluated :=
  match b.product with
  | some (x,y) => ⟨n.fma x y a.value,none⟩
  | none => match a.product with
    | some (x,y) => ⟨n.fma x y b.value,none⟩
    | none => ⟨n.add a.value b.value,none⟩

def sub (n : Numerics) (a b : Evaluated) : Evaluated :=
  match b.product with
  | some (x,y) => ⟨n.fma (n.neg x) y a.value,none⟩
  | none => ⟨n.sub a.value b.value,none⟩

/-- Local contraction profile for the selected default-fusion source. Its
compiler-profile binding is an outstanding local obligation; this definition
alone does not claim native numerical conformance. -/
def evaluate (p : ReductionProfile) (n : Numerics) : Expr → Evaluated
  | .raw x => ⟨x,none⟩
  | .literal x => ⟨n.literal x,none⟩
  | .natural x => ⟨n.natural x,none⟩
  | .negativeInfinity => ⟨n.negativeInfinity,none⟩
  | .barrier x => ⟨(evaluate p n x).value,none⟩
  | .binary op a b =>
    let left := evaluate p n a
    let right := evaluate p n b
    match op with
    | .add => add n left right
    | .sub => sub n left right
    | .mul => ⟨n.mul left.value right.value,some (left.value,right.value)⟩
    | .div => ⟨n.div left.value right.value,none⟩
  | .unary op x =>
    let value := (evaluate p n x).value
    match op with
    | .exp => ⟨n.exp value,none⟩
    | .log => ⟨n.log value,none⟩
  | .sum width values => ⟨(p.sumGraph width).eval n (fun i => (evaluate p n (values i)).value),none⟩
  | .maximum width values => ⟨(p.maximumGraph width).eval n (fun i => (evaluate p n (values i)).value),none⟩
  | .negation x => ⟨n.neg (evaluate p n x).value,none⟩
  | .hostAdd a b => ⟨n.hostAdd (evaluate p n a).value (evaluate p n b).value,none⟩
  | .hostSquare x => ⟨n.hostSquare (evaluate p n x).value,none⟩
  | .hostScale scale x => ⟨n.hostScale scale (evaluate p n x).value,none⟩
  | .hostSum width values => ⟨(p.hostSumGraph width).eval n (fun i => (evaluate p n (values i)).value),none⟩
  | .hostLSE width values => ⟨(p.hostLSEGraph width).eval n (fun i => (evaluate p n (values i)).value),none⟩
  | .rankSum width values => ⟨(p.rankSumGraph width).eval n (fun i => (evaluate p n (values i)).value),none⟩

def word (p : ReductionProfile) (n : Numerics) (expression : Expr) : Word := (evaluate p n expression).value

theorem add_equiv (h : Obligations n r) (a b : Evaluated) : add n a b=add r a b := by
  rcases a with ⟨av,ap⟩
  rcases b with ⟨bv,bp⟩
  cases bp with
  | some pair => rcases pair with ⟨x,y⟩; simp only [add,h.fma]
  | none =>
    cases ap with
    | some pair => rcases pair with ⟨x,y⟩; simp only [add,h.fma]
    | none => simp only [add,h.add]

theorem sub_equiv (h : Obligations n r) (a b : Evaluated) : sub n a b=sub r a b := by
  rcases b with ⟨bv,bp⟩
  cases bp with
  | some pair => rcases pair with ⟨x,y⟩; simp only [sub,h.neg,h.fma]
  | none => simp only [sub,h.sub]

/-- The entire source graph is proved by induction using scalar identities
and explicit selected reduction graphs. Numerical records are never equated. -/
theorem evaluate_equiv (p : ReductionProfile) (native reference : Numerics)
    (h : Obligations native reference) (expression : Expr) :
    evaluate p native expression=evaluate p reference expression := by
  induction expression with
  | raw => rfl
  | literal x => simp only [evaluate,h.literal]
  | natural x => simp only [evaluate,h.natural]
  | negativeInfinity => simp only [evaluate,h.negativeInfinity]
  | barrier x ih => simp only [evaluate,ih]
  | binary op a b ia ib =>
    cases op <;> simp only [evaluate,ia,ib,add_equiv h,sub_equiv h,h.mul,h.div]
  | unary op x ih => cases op <;> simp only [evaluate,ih,h.exp,h.log]
  | sum width xs ih | maximum width xs ih | hostSum width xs ih | hostLSE width xs ih | rankSum width xs ih =>
    apply congrArg (fun v => Evaluated.mk v none)
    apply ScalarGraph.equiv h
    intro i
    exact congrArg Evaluated.value (ih i)
  | negation x ih => simp only [evaluate,ih,h.neg]
  | hostAdd a b ia ib => simp only [evaluate,ia,ib,h.hostAdd]
  | hostSquare x ih => simp only [evaluate,ih,h.hostSquare]
  | hostScale scale x ih => simp only [evaluate,ih,h.hostScale]

theorem evaluation_equiv (p : ReductionProfile) (native reference : Numerics)
    (h : Obligations native reference) (expression : Expr) :
    word p native expression=word p reference expression :=
  congrArg Evaluated.value (evaluate_equiv p native reference h expression)

/-- A load barrier prevents a multiply from being contracted with a later
kernel's add. This is independent of every numerical value and special case. -/
theorem barrier_stops_contraction (p : ReductionProfile) (n : Numerics) (x : Expr) : (evaluate p n (.barrier x)).product=none := rfl

theorem fused_correction (p : ReductionProfile) (n : Numerics) (factor probability : Word) :
    word p n (.binary .add (.raw probability) (.binary .mul (.raw factor) (.raw probability)))=
      n.fma factor probability probability := rfl

theorem host_add_stops_contraction (p : ReductionProfile) (n : Numerics) (a b : Expr) :
    (evaluate p n (hostArithmetic.add a b)).product=none := rfl

theorem host_scale_stops_contraction (p : ReductionProfile) (n : Numerics) (scale : ℝ) (a : Expr) :
    (evaluate p n (hostArithmetic.scale scale a)).product=none := rfl

#axiomsClean evaluation_equiv
#axiomsClean barrier_stops_contraction
#axiomsClean fused_correction
#axiomsClean host_add_stops_contraction
#axiomsClean host_scale_stops_contraction
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating

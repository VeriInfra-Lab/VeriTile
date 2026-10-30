import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Triton.Float.Erasure

/-! Exact bool-count and signed-integer argmax: source lowering, arbitrary
axes, success/failure in every evaluator, empty shapes, and stable ties. -/
namespace VeriTile.Bench.Tests.DiscreteControls
open VeriTile.Triton

private def body : ComputeKernel → List ComputeStmt
  | .mk _ _ ss => ss

private def countOp? : {dtype : TileDType} → {shape : TileShape} →
    Op dtype shape → Option (TileShape × Nat × Bool)
  | _, _, .reduceSumBool (shape := sh) ax kd _ =>
      some (TileShape.reduceShape sh ax kd, ax.val, kd)
  | _, _, _ => none

private def argMaxOp? : {dtype : TileDType} → {shape : TileShape} →
    Op dtype shape → Option (TileShape × Nat)
  | _, _, .argMaxInt (shape := sh) ax _ => some (TileShape.eraseAxis sh ax, ax.val)
  | _, _, _ => none

private def lastCount? (k : ComputeKernel) : Option (TileShape × Nat × Bool) :=
  match (body k).reverse.head? with
  | some (.assign _ _ _ (.alg e)) => countOp? e
  | _ => none

private def lastArgMax? (k : ComputeKernel) : Option (TileShape × Nat) :=
  match (body k).reverse.head? with
  | some (.assign _ _ _ (.alg e)) => argMaxOp? e
  | _ => none

def rowCounts : ComputeKernel := triton {
  flags = tl.arange(0, 2)[:, None] < tl.arange(0, 4)[None, :]
  counts = tl.sum(flags, 1)
}

def columnCounts : ComputeKernel := triton {
  flags = tl.arange(0, 2)[:, None] < tl.arange(0, 4)[None, :]
  counts = tl.sum(flags, axis=0, keep_dims=true)
}

def oneCount : ComputeKernel := triton {
  flags = tl.arange(0, 4) < 2
  count = tl.sum(flags)
}

def signedIndices (input : Region .int) : ComputeKernel := triton {
  tl.region input = tl.int64
  values = tl.load(input + tl.arange(0, 4))
  indices = tl.argmax(values, 0, tie_break_left=True)
}

def signed32Indices (input : Region .int) : ComputeKernel := triton {
  tl.region input = tl.int32
  values = tl.load(input + tl.arange(0, 4))
  indices = tl.argmax(values, 0)
}

def signed8Indices (input : Region .int) : ComputeKernel := triton {
  tl.region input = tl.int8
  values = tl.load(input + tl.arange(0, 4))
  indices = tl.argmax(values, 0)
}

theorem signed32_argmax_shape (input : Region .int) :
    lastArgMax? (signed32Indices input) = some ([],0) := rfl

theorem signed8_argmax_shape (input : Region .int) :
    lastArgMax? (signed8Indices input) = some ([],0) := rfl

theorem row_count_shape : lastCount? rowCounts = some ([2], 1, Bool.false) := rfl
theorem column_count_shape : lastCount? columnCounts = some ([1,4], 0, Bool.true) := rfl
theorem rank_one_implicit_axis : lastCount? oneCount = some ([], 0, Bool.false) := rfl
theorem signed_argmax_shape (input : Region .int) :
    lastArgMax? (signedIndices input) = some ([],0) := rfl

/-- error: tl.sum: bool inputs support only the default uint32 reduction dtype -/
#guard_msgs in
def unsupportedExplicitDType : ComputeKernel := triton {
  flags = tl.arange(0, 4) < 2
  count = tl.sum(flags, 0, dtype=tl.int32)
}

/-- error: tl.argmax: only literal True leftmost tie breaking is supported -/
#guard_msgs in
def unsupportedTie (input : Region .int) : ComputeKernel := triton {
  tl.region input = tl.int32
  values = tl.load(input + tl.arange(0, 4))
  indices = tl.argmax(values, 0, tie_break_left=False)
}

/-- error: tl.argmax: signed integer tiles of rank > 1 require an explicit axis -/
#guard_msgs in
def unsupportedImplicitAxis (input : Region .int) : ComputeKernel := triton {
  tl.region input = tl.int32
  values = tl.load(input + tl.arange(0, 4))
  indices = tl.argmax(values[None, :])
}

theorem count_range {n : Nat} (flags : Fin n → Bool) :
    TileShape.boolCount32 flags < 2^32 := Nat.mod_lt _ (by decide)

theorem count_exact {n : Nat} (flags : Fin n → Bool) (hn : n < 2^32) :
    TileShape.boolCount32 flags = ∑ i, if flags i then (1 : Nat) else 0 := by
  apply Nat.mod_eq_of_lt
  apply lt_of_le_of_lt _ hn
  calc
    _ ≤ ∑ _i : Fin n, 1 := Finset.sum_le_sum (fun i _ => by split <;> omega)
    _ = n := by simp

theorem wrapper_count_exact (flags : Fin 128 → Bool) :
    TileShape.boolCount32 flags = ∑ i, if flags i then (1 : Nat) else 0 :=
  count_exact flags (by decide)

theorem count_all_false (n : Nat) : TileShape.boolCount32 (fun _ : Fin n => Bool.false) = 0 := by
  simp [TileShape.boolCount32]

theorem count_all_true : TileShape.boolCount32 (fun _ : Fin 128 => Bool.true) = 128 := by
  simp [TileShape.boolCount32]

theorem count_empty (flags : Fin 0 → Bool) : TileShape.boolCount32 flags = 0 := by
  simp [TileShape.boolCount32]

theorem int_argmax_negative :
    TileShape.leftmostIntArgMax (fun i : Fin 4 =>
      if i.val = 0 then -5 else if i.val = 1 then -2 else -3) = 1 := by decide

theorem int_argmax_tie :
    TileShape.leftmostIntArgMax (fun i : Fin 4 =>
      if i.val = 0 then -5 else if i.val < 3 then 7 else -1) = 1 := by decide

theorem int_argmax_empty (values : Fin 0 → Int) : TileShape.leftmostIntArgMax values = 0 := rfl

theorem int_argmax_bound {n : Nat} (values : Fin n → Int) (hn : 0 < n) :
    TileShape.leftmostIntArgMax values < n := by
  simp only [TileShape.leftmostIntArgMax, hn, ↓reduceDIte]
  exact Fin.isLt _

theorem row_count_values :
    TileShape.discreteBoolSum (shape := [2,4]) ⟨1, by decide⟩ Bool.false
      (fun i => decide (i.1.val < i.2.1.val)) =
        fun i => if i.1.val = 0 then 3 else 2 := by
  funext i
  rcases i with ⟨r,⟨⟩⟩
  fin_cases r <;> decide

theorem column_count_values :
    TileShape.discreteBoolSum (shape := [2,4]) ⟨0, by decide⟩ Bool.true
      (fun i => decide (i.1.val < i.2.1.val)) =
        fun i => if i.2.1.val = 0 then 0 else if i.2.1.val = 1 then 1 else 2 := by
  funext i
  rcases i with ⟨r,c,⟨⟩⟩
  fin_cases r
  fin_cases c <;> decide

theorem real_count_success {sh : TileShape} (ax : Fin sh.length) (kd : Bool)
     (e : Op .bool sh) (s : BlockState) (v : Tile .bool sh)
    (h : evalOp e s = some v) :
    evalOp (Op.reduceSumBool ax kd e) s = some ⟨TileShape.discreteBoolSum ax kd v.data⟩ := by
  simp only [evalOp, h, bind, Option.bind, Option.pure_def]

theorem real_count_failure {sh : TileShape} (ax : Fin sh.length) (kd : Bool)
     (e : Op .bool sh) (s : BlockState) (h : evalOp e s = none) :
    evalOp (Op.reduceSumBool ax kd e) s = none := by
  simp only [evalOp, h, bind, Option.bind, Option.pure_def]

theorem rounded_count_success {sh : TileShape} (ax : Fin sh.length) (kd : Bool)
    (R : RoundingModel) (e : Op .bool sh) (s : BlockState) (v : Tile .bool sh)
    (h : evalOpR R e s = some v) :
    evalOpR R (Op.reduceSumBool ax kd e) s = some ⟨TileShape.discreteBoolSum ax kd v.data⟩ := by
  simp only [evalOpR, h, bind, Option.bind, Option.pure_def]

theorem rounded_count_failure {sh : TileShape} (ax : Fin sh.length) (kd : Bool)
    (R : RoundingModel) (e : Op .bool sh) (s : BlockState) (h : evalOpR R e s = none) :
    evalOpR R (Op.reduceSumBool ax kd e) s = none := by
  simp only [evalOpR, h, bind, Option.bind, Option.pure_def]

theorem typed_count_success {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (ax : Fin sh.length) (kd : Bool)
    (e : Op .bool sh) (s : FP.Structural.State α) (v : FP.Structural.Values α .bool sh)
    (h : FP.Structural.evalOp M p e s = some v) :
    FP.Structural.evalOp M p (Op.reduceSumBool ax kd e) s = some (TileShape.discreteBoolSum ax kd v) := by
  rw [FP.Structural.evalOp_unfold]
  simp only [h, bind, Option.bind, Option.pure_def]

theorem typed_count_failure {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (ax : Fin sh.length) (kd : Bool)
    (e : Op .bool sh) (s : FP.Structural.State α)
    (h : FP.Structural.evalOp M p e s = none) :
    FP.Structural.evalOp M p (Op.reduceSumBool ax kd e) s = none := by
  rw [FP.Structural.evalOp_unfold]
  simp only [h, bind, Option.bind, Option.pure_def]

theorem erase_count {sh : TileShape} (ax : Fin sh.length) (kd : Bool)
    (e : Op .bool sh) : (Op.reduceSumBool ax kd e).eraseDType = Op.reduceSumBool ax kd e.eraseDType := by
  rw [Op.eraseDType.eq_def]

theorem real_argmax_success {sh : TileShape} (ax : Fin sh.length) 
     (e : Op .int sh) (s : BlockState) (v : Tile .int sh)
    (h : evalOp e s = some v) :
    evalOp (Op.argMaxInt ax e) s = some ⟨TileShape.discreteIntArgMax ax v.data⟩ := by
  simp only [evalOp, h, bind, Option.bind, Option.pure_def]

theorem real_argmax_failure {sh : TileShape} (ax : Fin sh.length) 
     (e : Op .int sh) (s : BlockState) (h : evalOp e s = none) :
    evalOp (Op.argMaxInt ax e) s = none := by
  simp only [evalOp, h, bind, Option.bind, Option.pure_def]

theorem rounded_argmax_success {sh : TileShape} (ax : Fin sh.length) 
    (R : RoundingModel) (e : Op .int sh) (s : BlockState) (v : Tile .int sh)
    (h : evalOpR R e s = some v) :
    evalOpR R (Op.argMaxInt ax e) s = some ⟨TileShape.discreteIntArgMax ax v.data⟩ := by
  simp only [evalOpR, h, bind, Option.bind, Option.pure_def]

theorem rounded_argmax_failure {sh : TileShape} (ax : Fin sh.length) 
    (R : RoundingModel) (e : Op .int sh) (s : BlockState) (h : evalOpR R e s = none) :
    evalOpR R (Op.argMaxInt ax e) s = none := by
  simp only [evalOpR, h, bind, Option.bind, Option.pure_def]

theorem typed_argmax_success {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (ax : Fin sh.length) 
    (e : Op .int sh) (s : FP.Structural.State α) (v : FP.Structural.Values α .int sh)
    (h : FP.Structural.evalOp M p e s = some v) :
    FP.Structural.evalOp M p (Op.argMaxInt ax e) s = some (TileShape.discreteIntArgMax ax v) := by
  rw [FP.Structural.evalOp_unfold]
  simp only [h, bind, Option.bind, Option.pure_def]

theorem typed_argmax_failure {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (ax : Fin sh.length) 
    (e : Op .int sh) (s : FP.Structural.State α)
    (h : FP.Structural.evalOp M p e s = none) :
    FP.Structural.evalOp M p (Op.argMaxInt ax e) s = none := by
  rw [FP.Structural.evalOp_unfold]
  simp only [h, bind, Option.bind, Option.pure_def]

theorem erase_argmax {sh : TileShape} (ax : Fin sh.length) 
    (e : Op .int sh) : (Op.argMaxInt ax e).eraseDType = Op.argMaxInt ax e.eraseDType := by
  rw [Op.eraseDType.eq_def]

theorem typed_empty_count_child_failure {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (.reduceSumBool (shape := [0,4]) ⟨1,by decide⟩ Bool.false
      (.ref .bool [0,4] "missing")) {s with regs := fun _ _ _ => none} = none := by
  apply typed_count_failure
  rw [FP.Structural.evalOp_unfold]

theorem typed_empty_argmax_child_failure {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (.argMaxInt (shape := [0,4]) ⟨1,by decide⟩
      (.ref .int [0,4] "missing")) {s with regs := fun _ _ _ => none} = none := by
  apply typed_argmax_failure
  rw [FP.Structural.evalOp_unfold]

theorem first_true_fold {α : Type} (flags : α → Bool) (indices : List α) (initial : α) :
    indices.foldl (fun best candidate =>
      if (if flags best then (1 : Int) else 0) < (if flags candidate then 1 else 0)
      then candidate else best) initial =
      if flags initial then initial else (indices.find? flags).getD initial := by
  induction indices generalizing initial with
  | nil => simp
  | cons i is ih =>
      simp only [List.foldl_cons]
      rw [ih]
      cases hi : flags i <;> cases hb : flags initial <;> simp [hi, hb]

theorem argmax_flags_first {n : Nat} (flags : Fin n → Bool) (hn : 0 < n) :
    TileShape.leftmostIntArgMax (fun i => if flags i then (1 : Int) else 0) =
      if flags ⟨0,hn⟩ then 0 else
        (((List.finRange n).find? flags).getD ⟨0,hn⟩).val := by
  simp only [TileShape.leftmostIntArgMax, hn, ↓reduceDIte]
  rw [first_true_fold]
  cases flags ⟨0,hn⟩ <;> rfl

theorem empty_axis_count :
    TileShape.discreteBoolSum (shape := [2,0]) ⟨1,by decide⟩ Bool.false
      (fun _ => Bool.true) = fun _ => 0 := by
  funext i
  exact count_empty _

theorem empty_axis_argmax :
    TileShape.discreteIntArgMax (shape := [2,0]) ⟨1,by decide⟩
      (fun _ => -1) = fun _ => 0 := rfl

theorem typed_count_register_success {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (v : FP.Structural.Values α .bool [2,4]) :
    FP.Structural.evalOp M p (.reduceSumBool (shape := [2,4]) ⟨1,by decide⟩ Bool.false
      (.ref .bool [2,4] "flags")) (s.setReg "flags" .bool [2,4] v) =
      some (TileShape.discreteBoolSum ⟨1,by decide⟩ Bool.false v) := by
  apply typed_count_success
  simp only [FP.Structural.evalOp_unfold, FP.Structural.State.setReg_same]

theorem typed_argmax_register_success {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (s : FP.Structural.State α) (v : FP.Structural.Values α .int [2,4]) :
    FP.Structural.evalOp M p (.argMaxInt (shape := [2,4]) ⟨1,by decide⟩
      (.ref .int [2,4] "values")) (s.setReg "values" .int [2,4] v) =
      some (TileShape.discreteIntArgMax ⟨1,by decide⟩ v) := by
  apply typed_argmax_success
  simp only [FP.Structural.evalOp_unfold, FP.Structural.State.setReg_same]

#print axioms signed32_argmax_shape
#print axioms signed8_argmax_shape
#print axioms row_count_shape
#print axioms column_count_shape
#print axioms rank_one_implicit_axis
#print axioms signed_argmax_shape
#print axioms count_range
#print axioms count_exact
#print axioms wrapper_count_exact
#print axioms count_all_false
#print axioms count_all_true
#print axioms count_empty
#print axioms int_argmax_negative
#print axioms int_argmax_tie
#print axioms int_argmax_empty
#print axioms int_argmax_bound
#print axioms row_count_values
#print axioms column_count_values
#print axioms real_count_success
#print axioms real_count_failure
#print axioms rounded_count_success
#print axioms rounded_count_failure
#print axioms typed_count_success
#print axioms typed_count_failure
#print axioms erase_count
#print axioms real_argmax_success
#print axioms real_argmax_failure
#print axioms rounded_argmax_success
#print axioms rounded_argmax_failure
#print axioms typed_argmax_success
#print axioms typed_argmax_failure
#print axioms erase_argmax
#print axioms typed_empty_count_child_failure
#print axioms typed_empty_argmax_child_failure

#print axioms first_true_fold

#print axioms argmax_flags_first

#print axioms empty_axis_count

#print axioms empty_axis_argmax

#print axioms typed_count_register_success

#print axioms typed_argmax_register_success

end VeriTile.Bench.Tests.DiscreteControls

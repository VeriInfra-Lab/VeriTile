import bench.tritonbench_g.flash_decode2_phi.Proofs.KernelTensor
import bench.tritonbench_g.flash_decode2_phi.Proofs.Numerical

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelNative
open VeriTile.Triton FP.Structural FlashDecode2 Native
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

abbrev Ops := Numerical.ScalarOps
abbrev lower := Numerical.lower

/-- Independent source-ordered primitive graph, including load boundaries.
Compiler fusion/order choices see shapes, never raw numerical payloads. -/
structure Accumulator where
  maximum : Term
  denominator : Term
  numerator : Nat → Term

def initial : Accumulator := ⟨.negInf,.literal none .real 0,fun _ => .literal (some .fp32) .real 0⟩
def op (o : Binary) (a b : Term) : Term := .binary (some .fp32) .real o a b
def exp (a : Term) : Term := .unary (some .fp32) .exp a

def merge (x : Term) (v : Nat → Term) (a : Accumulator) : Accumulator :=
  let m := op .max x a.maximum
  let old := exp (op .sub a.maximum m)
  let fresh := exp (op .sub x m)
  ⟨m,op .add (op .mul a.denominator old) fresh,
    fun d => op .add (op .mul (a.numerator d) old) (op .mul fresh (v d))⟩
def scan (x : Nat → Term) (v : Nat → Nat → Term) : Nat → Accumulator
  | 0 => initial
  | n+1 => merge (x n) (v n) (scan x v n)

def logs (c : Call) (heap : KernelTensor.Heap) (row j : Nat) : Term :=
  let i := row*Checked.capacity c.val+j
  .load (.leaf "E" i (KernelTensor.word c heap "E" i))
def mids (c : Call) (heap : KernelTensor.Heap) (row j d : Nat) : Term :=
  let i := row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+d
  .load (if d < Checked.width c.val then .leaf "M" i (KernelTensor.word c heap "M" i)
    else .literal (some .fp32) .real 0)

def term (c : Call) (heap : KernelTensor.Heap) (row d : Nat) : Term :=
  let a := scan (logs c heap row) (mids c heap row)
    (KernelModel.iterations c (KernelTensor.lengths c heap) row)
  op .div (a.numerator d) a.denominator

def target (p : Ops) (S : Schedule) (c : Call) (heap : KernelTensor.Heap)
    (row d : Nat) : Word := eval (lower p) S (term c heap row d)

def entry (c : Call) (heap : KernelTensor.Heap) (other : State Term) :=
  KernelTensor.lift Term.leaf c heap other

theorem log_eq (c : Call) (heap : KernelTensor.Heap) (other : State Term) (row j : Nat) :
    logvalue algebra (Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other))
      (Checked.heads c.val) (Checked.capacity c.val) j=logs c heap row j := by
  simp only [logvalue,row_log_offset]
  simp [algebra,Grid.setup,entry,KernelTensor.lift,Cell.read_same,logs]

theorem mid_eq (c : Call) (heap : KernelTensor.Heap) (other : State Term) (row j : Nat)
    (k : TileIndex [Checked.block c.val]) :
    FlashDecode2.value algebra (Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other))
      c.val.masked (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) j k=
      mids c heap row j (lane k) := by
  simp only [FlashDecode2.value,row_mid_offset]
  by_cases live : lane k<Checked.width c.val <;>
    simp [algebra,Grid.setup,entry,KernelTensor.lift,Cell.read_same,mids,active,c.property,live,zero]

theorem recurrence (c : Call) (heap : KernelTensor.Heap) (other : State Term)
    (row n : Nat) :
    let s := Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other)
    let a := scan (logs c heap row) (mids c heap row) n
    (values algebra s c.val.masked (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) (Checked.block c.val) n).m = a.maximum ∧
    (values algebra s c.val.masked (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) (Checked.block c.val) n).z = a.denominator ∧
    ∀ k,(values algebra s c.val.masked (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) (Checked.block c.val) n).a k = a.numerator (lane k) := by
  induction n with
  | zero => simp [values,FlashDecode2.initial,zero,algebra,scan,initial]
  | succ n ih =>
    dsimp only at ih ⊢
    simp only [values,next,ih.1,ih.2.1,ih.2.2,log_eq,mid_eq]
    simp only [maximum,exponential,sub,FlashDecode2.mul,add,algebra,scan,merge,op,exp,
      and_self,implies_true]

theorem value (p : Ops) (S : Schedule) (c : Call) (heap : KernelTensor.Heap) (other : State Term)
    (h : InputContract c (KernelTensor.lengths c heap)) (row : Nat) (hr : row < KernelModel.rows c)
    (k : TileIndex [Checked.block c.val]) :
    eval (lower p) S (Grid.value algebra (entry c heap other) c.val.masked
      (Checked.batches c.val) (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) c.val.blockSeq (Checked.block c.val) row k)=
      target p S c heap row (lane k) := by
  have counted := KernelModel.count_bound c _ h (entry c heap other)
    (KernelTensor.lift_length _ c heap other) row hr
  have recur := recurrence c heap other row (KernelModel.iterations c (KernelTensor.lengths c heap) row)
  simp only [Grid.value,output,counted.1]
  rw [recur.2.2 k,recur.2.1]
  rfl

#axiomsClean recurrence
#axiomsClean value
#specNonCircular target avoiding [KernelModel.run,Grid.source,FP.Structural.exec,values]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelNative

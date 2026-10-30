import bench.repairs.tritonbench_g.bmm_optimized.Whole

namespace VeriTile.Bench.Repairs.BmmOptimized.Checked
open Host Native

/-- Rank-three strided FP32 CUDA and compatible batch/K classification is
performed before this descriptor. Nat strides permit broadcasts. Origins are
actual view data pointers, so storage offsets require no exclusion. -/
structure Metadata where
  call : Call
  pointerA : Nat
  pointerB : Nat

def metadataChecks (p : Metadata) : Bool :=
  decide (p.call.a.device=p.call.b.device ∧ p.call.a.negative=false ∧ p.call.b.negative=false ∧
    (p.call.shape.batch*p.call.shape.m*p.call.shape.k=0 ∨ p.pointerA%4=0) ∧
    (p.call.shape.batch*p.call.shape.k*p.call.shape.n=0 ∨ p.pointerB%4=0))

def indexChecks (s : Shape) : Bool :=
  decide (s.batch<2^31 ∧ s.m+128<2^31 ∧ s.n+128<2^31 ∧ s.k+64<2^31 ∧
    2*blocks s.m 32*blocks s.n 32+2<2^31 ∧
    (s.batch*s.m+128)*s.k+64<2^31 ∧ (s.batch*s.k+64)*s.n+128<2^31 ∧
    (s.batch*s.m+128)*s.n+128<2^31)

inductive Error where | dynamic | metadata | indices deriving DecidableEq, Repr

def checkInputs (p : Metadata) : Except Error Unit :=
  if !metadataChecks p then .error .metadata
  else if !indexChecks p.call.shape then .error .indices else .ok ()

structure InputContract (p : Metadata) : Prop where
  metadata : metadataChecks p=true
  indices : IndexABI p.call.shape

theorem indexChecks_iff (s : Shape) : indexChecks s=true ↔ IndexABI s := by
  simp only [indexChecks,decide_eq_true_eq]
  constructor
  · rintro ⟨b,m,n,k,g,a,c,o⟩; exact ⟨b,m,n,k,g,a,c,o⟩
  · intro h; exact ⟨h.batch,h.rows,h.columns,h.contraction,h.grid,h.a,h.b,h.output⟩

theorem checkInputs_iff (p : Metadata) : checkInputs p=.ok () ↔ InputContract p := by
  have h : InputContract p ↔ metadataChecks p=true ∧ indexChecks p.call.shape=true := by
    constructor
    · intro h; exact ⟨h.metadata,(indexChecks_iff _).2 h.indices⟩
    · rintro ⟨m,i⟩; exact ⟨m,(indexChecks_iff _).1 i⟩
  rw [h]
  simp [checkInputs,ite_eq_iff]

inductive Arguments where | unsupported | tensors (p : Metadata)
def checkArguments : Arguments → Except Error Unit
  | .unsupported => .error .dynamic
  | .tensors p => checkInputs p
def AdmittedArguments : Arguments → Prop
  | .unsupported => False
  | .tensors p => InputContract p
theorem checkArguments_iff (args : Arguments) : checkArguments args=.ok () ↔ AdmittedArguments args := by
  cases args <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

/-- Valid storage supplies the covered span even for broadcasts and aliases.
Fresh output identity follows from the existing torch.empty allocation. -/
structure Ordinary (call : Call) : Prop where
  aSpan : 0<call.shape.batch → 0<call.shape.m → 0<call.shape.k →
    (call.shape.batch-1)*call.a.batchStride+(call.shape.m-1)*call.a.rowStride+
      (call.shape.k-1)*call.a.columnStride<call.a.storageWords
  bSpan : 0<call.shape.batch → 0<call.shape.k → 0<call.shape.n →
    (call.shape.batch-1)*call.b.batchStride+(call.shape.k-1)*call.b.rowStride+
      (call.shape.n-1)*call.b.columnStride<call.b.storageWords
  freshA : call.freshIdentity≠call.a.identity
  freshB : call.freshIdentity≠call.b.identity

theorem covered (v : View) (batch rows cols b row col : Nat)
    (span : 0<batch → 0<rows → 0<cols →
      (batch-1)*v.batchStride+(rows-1)*v.rowStride+(cols-1)*v.columnStride<v.storageWords)
    (hb : b<batch) (hr : row<rows) (hc : col<cols) : v.address b row col<v.storageWords := by
  have h1 := Nat.mul_le_mul_right v.batchStride (show b≤batch-1 by omega)
  have h2 := Nat.mul_le_mul_right v.rowStride (show row≤rows-1 by omega)
  have h3 := Nat.mul_le_mul_right v.columnStride (show col≤cols-1 by omega)
  have bound := span (by omega) (by omega) (by omega)
  unfold View.address
  omega

theorem raw (p : Metadata) (admitted : InputContract p) :
    p.call.a.negative=false ∧ p.call.b.negative=false := by
  have h := admitted.metadata
  simp only [metadataChecks,decide_eq_true_eq] at h
  exact ⟨h.2.1,h.2.2.1⟩

/-- Existing contiguous calls already have the raw logical preparation below;
there is no additional resolution/copy primitive in this checked revision. -/
theorem prepare_raw {α : Type} (p : Metadata) (admitted : InputContract p)
    (neg : α → α) (m : Memory α) : prepare neg p.call m=prepare id p.call m := by
  funext region address
  have h := raw p admitted
  cases region <;> simp [prepare,packedRead,logical,h.1,h.2]

theorem chain_raw_equiv (p : Metadata) (admitted : InputContract p)
    (actual target : Instructions)
    (fma : ∀ a b carry, actual.fmaRN a b carry=target.fmaRN a b carry)
    (m : Memory Word) (x : Cell) (n : Nat) :
    chain actual p.call m x n=chain target p.call m x n := by
  have h := raw p admitted
  induction n with
  | zero => rfl
  | succ n ih => simp only [chain,logical,h.1,h.2,Bool.false_eq_true,ite_false,ih,fma]

#axiomsClean checkArguments_iff
#axiomsClean covered
#axiomsClean prepare_raw
#axiomsClean chain_raw_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmOptimized.Checked

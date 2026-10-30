import VeriTile.Triton.Memory.LogicalIO
import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference
import bench.tritonbench_g.diag_ssm_triton.Proofs.Erasure
import VeriTile.Triton.Memory.RegionalKernelIO

/-! Typed finite windows for all four per-program time scans. Complex channels
split the real and imaginary components of each logical input/output role.
The shared regional judgment allows coherent read-only aliases. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section

theorem erased_projects (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgorithm? = .ok k.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def components (complex : Bool) : Nat := if complex then 2 else 1

def column (B pid lane : Nat) : Nat := pid*B+lane

def streamAddress (batch dim B pid k : Nat) : Nat :=
  (k/B)*(batch*dim)+column B pid (k%B)

def active (batch dim B pid lane : Nat) : Prop := column B pid lane < batch*dim

def forwardArity (B length : Nat) (role : Fin 3) : Nat :=
  if role.val < 2 then B else length*B

def forwardBuffer (p : ForwardLayout) : Fin 3 → RegionName
  | ⟨0,_⟩ => p.s | ⟨1,_⟩ => p.l | ⟨2,_⟩ => p.x

def forwardAddress (p : ForwardLayout) (pid : Nat) (role : Fin 3) (k : Nat) : Nat :=
  match role with
  | ⟨0,_⟩ => column p.B pid k
  | ⟨1,_⟩ => column p.B pid k % p.dim
  | ⟨2,_⟩ => streamAddress p.batch p.dim p.B pid k

def forwardLane (B : Nat) (role : Fin 3) (k : Nat) : Nat :=
  if role.val < 2 then k else k%B

def forwardKernel (complex : Bool) (p : ForwardLayout) (length : Nat) : ComputeKernel :=
  if complex then
    (Kernels.complexForwardTyped p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType
  else (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType

def forwardIO (complex : Bool) (p : ForwardLayout) (length : Nat) : UKernelIO where
  kernel := forwardKernel complex p length
  projection := by
    cases complex <;> simp only [forwardKernel,Bool.false_eq_true,↓reduceIte]
    all_goals apply erased_projects; rfl
  nIn := 3*components complex
  nOut := components complex
  nScr := 0
  bufs := [p.s,p.l,p.x,p.y]
  ity := fun _ => .float
  iarity := fun i => forwardArity p.B length i.divNat
  ibuf := fun i => forwardBuffer p i.divNat
  oty := fun _ => .float
  oarity := fun _ => length*p.B
  obuf := fun _ => p.y
  obuf_mem := fun _ => by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i _ pid _ _ k => forwardAddress p pid i.divNat k.val * components complex + i.modNat.val
  imask := fun i _ pid _ _ k => active p.batch p.dim p.B pid (forwardLane p.B i.divNat k.val)
  owin := fun o _ pid _ _ k => streamAddress p.batch p.dim p.B pid k.val * components complex + o.val
  omask := fun _ _ pid _ _ k => active p.batch p.dim p.B pid (k.val%p.B)
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i

def backwardArity (B length : Nat) (role : Fin 4) : Nat :=
  if role.val < 2 then B else if role.val = 2 then (length-1)*B else length*B

def backwardBuffer (p : BackwardLayout) : Fin 4 → RegionName
  | ⟨0,_⟩ => p.s | ⟨1,_⟩ => p.l | ⟨2,_⟩ => p.y | ⟨3,_⟩ => p.gy

def backwardAddress (p : BackwardLayout) (pid : Nat) (role : Fin 4) (k : Nat) : Nat :=
  match role with
  | ⟨0,_⟩ => column p.B pid k
  | ⟨1,_⟩ => column p.B pid k % p.dim
  | ⟨2,_⟩ | ⟨3,_⟩ => streamAddress p.batch p.dim p.B pid k

def backwardLane (B : Nat) (role : Fin 4) (k : Nat) : Nat :=
  if role.val < 2 then k else k%B

def backwardOutArity (B length : Nat) (role : Fin 3) : Nat :=
  if role.val = 0 then length*B else B

def backwardOutBuffer (p : BackwardLayout) : Fin 3 → RegionName
  | ⟨0,_⟩ => p.gx | ⟨1,_⟩ => p.gs | ⟨2,_⟩ => p.gl

def backwardOutAddress (p : BackwardLayout) (pid : Nat) (role : Fin 3) (k : Nat) : Nat :=
  if role.val = 0 then streamAddress p.batch p.dim p.B pid k else column p.B pid k

def backwardOutLane (B : Nat) (role : Fin 3) (k : Nat) : Nat :=
  if role.val = 0 then k%B else k

def backwardKernel (complex : Bool) (p : BackwardLayout) (length : Nat) : ComputeKernel :=
  if complex then
    (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
  else (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType

def backwardIO (complex : Bool) (p : BackwardLayout) (length : Nat) : UKernelIO where
  kernel := backwardKernel complex p length
  projection := by
    cases complex <;> simp only [backwardKernel,Bool.false_eq_true,↓reduceIte]
    all_goals apply erased_projects; rfl
  nIn := 4*components complex
  nOut := 3*components complex
  nScr := 0
  bufs := [p.s,p.l,p.y,p.gy,p.gx,p.gs,p.gl]
  ity := fun _ => .float
  iarity := fun i => backwardArity p.B length i.divNat
  ibuf := fun i => backwardBuffer p i.divNat
  oty := fun _ => .float
  oarity := fun o => backwardOutArity p.B length o.divNat
  obuf := fun o => backwardOutBuffer p o.divNat
  obuf_mem := by intro o; rcases o.divNat with ⟨i,h⟩; interval_cases i <;> simp [backwardOutBuffer]
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i _ pid _ _ k => backwardAddress p pid i.divNat k.val * components complex + i.modNat.val
  imask := fun i _ pid _ _ k =>
    (i.divNat.val = 0 → 0 < length) ∧ active p.batch p.dim p.B pid (backwardLane p.B i.divNat k.val)
  owin := fun o _ pid _ _ k => backwardOutAddress p pid o.divNat k.val * components complex + o.modNat.val
  omask := fun o _ pid _ _ k => active p.batch p.dim p.B pid (backwardOutLane p.B o.divNat k.val)
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i

/-- Finite logical arrays are extended only to give the independent recurrence
its total mathematical function type; no extra input storage is declared. -/
def extend {n : Nat} (x : Fin n → ℝ) (k : Nat) : ℝ := if h : k<n then x ⟨k,h⟩ else 0

def forwardRealInputs (p : ForwardLayout) (length : Nat) (v : (forwardIO Bool.false p length).Ctx)
    (role : Fin 3) (k : Nat) : ℝ := extend (v role) k

def forwardComplexInputs (p : ForwardLayout) (length : Nat) (v : (forwardIO Bool.true p length).Ctx)
    (role : Fin 3) (k : Nat) : ℂ :=
  ⟨extend (v ⟨2*role.val,by have := role.isLt; simp [forwardIO,components]; omega⟩) k,
   extend (v ⟨2*role.val+1,by have := role.isLt; simp [forwardIO,components]; omega⟩) k⟩

def backwardRealInputs (p : BackwardLayout) (length : Nat) (v : (backwardIO Bool.false p length).Ctx)
    (role : Fin 4) (k : Nat) : ℝ := extend (v role) k

def backwardComplexInputs (p : BackwardLayout) (length : Nat) (v : (backwardIO Bool.true p length).Ctx)
    (role : Fin 4) (k : Nat) : ℂ :=
  ⟨extend (v ⟨2*role.val,by have := role.isLt; simp [backwardIO,components]; omega⟩) k,
   extend (v ⟨2*role.val+1,by have := role.isLt; simp [backwardIO,components]; omega⟩) k⟩

def forwardFormula {α : Type} [Semiring α] (B : Nat) (v : Fin 3 → Nat → α) (k : Nat) : α :=
  Reference.forwardState (v 0 (k%B)) (v 1 (k%B)) (fun t => v 2 (t*B+k%B)) (k/B+1)

def backwardFormula {α : Type} [Semiring α] (conjugate : α → α) (B length : Nat)
    (v : Fin 4 → Nat → α) (role : Fin 3) (k : Nat) : α :=
  let lane := backwardOutLane B role k
  let l := conjugate (v 1 lane)
  let gy := fun t => v 3 (t*B+lane)
  match role with
  | ⟨0,_⟩ => Reference.inputGradient length l gy (k/B)
  | ⟨1,_⟩ => Reference.reverseCarry length l gy length
  | ⟨2,_⟩ => Reference.transitionGradient length l (conjugate (v 0 lane))
      (fun t => conjugate (v 2 (t*B+lane))) gy length

end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface

namespace VeriTile.Bench.Repairs.DiagSsmTriton.LogicalIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton Interface
noncomputable section

def matrix (A : Type) (rows cols : Nat) : (Fin (rows*cols) → A) ≃ (Fin rows → Fin cols → A) where
  toFun v row col := v (finProdFinEquiv (row,col))
  invFun v k := v k.divNat k.modNat
  left_inv := by intro v; funext k; exact congrArg v (finProdFinEquiv.apply_symm_apply k)
  right_inv := by
    intro v; funext row col
    change v ((finProdFinEquiv.symm (finProdFinEquiv (row,col))).1)
      ((finProdFinEquiv.symm (finProdFinEquiv (row,col))).2) = v row col
    rw [finProdFinEquiv.symm_apply_apply]

@[ext] structure ForwardInputs (A : Type) (B length : Nat) where
  initial : Fin B → A
  transition : Fin B → A
  input : Fin length → Fin B → A
@[ext] structure BackwardInputs (A : Type) (B length : Nat) where
  initial : Fin B → A
  transition : Fin B → A
  saved : Fin (length-1) → Fin B → A
  upstream : Fin length → Fin B → A
@[ext] structure BackwardOutputs (A : Type) (B length : Nat) where
  inputGradient : Fin length → Fin B → A
  initialGradient : Fin B → A
  transitionGradient : Fin B → A

/-- Total extension in logical time; no physical row stride occurs here. -/
def timeAt {A : Type} [Zero A] {length B : Nat} (x : Fin length → Fin B → A) (t : Nat) (lane : Fin B) : A :=
  if h : t<length then x ⟨t,h⟩ lane else 0

def forwardRealInputs (p : ForwardLayout) (length : Nat) :
    (forwardIO Bool.false p length).Ctx ≃ ForwardInputs ℝ p.B length where
  toFun v := ⟨(v (0 : Fin 3)),(v (1 : Fin 3)),matrix ℝ length p.B (v (2 : Fin 3))⟩
  invFun xs := Fin.cases xs.initial (Fin.cases xs.transition (Fin.cases ((matrix ℝ length p.B).symm xs.input) ((fun i => i.elim0))))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    · exact congrFun ((matrix ℝ length p.B).symm_apply_apply _) k
  right_inv := by
    intro xs
    apply ForwardInputs.ext
    · rfl
    · rfl
    · change (matrix ℝ length p.B) ((matrix ℝ length p.B).symm xs.input) = xs.input
      exact (matrix ℝ length p.B).apply_symm_apply _

def forwardRealOutputs (p : ForwardLayout) (length : Nat) :
    ((o : Fin (forwardIO Bool.false p length).nOut) → Fin ((forwardIO Bool.false p length).oarity o) → ((forwardIO Bool.false p length).oty o).carrier) ≃ (Fin length → Fin p.B → ℝ) where
  toFun v := matrix ℝ length p.B (v (0 : Fin 1))
  invFun xs := Fin.cases ((matrix ℝ length p.B).symm xs) ((fun i => i.elim0))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    exact congrFun ((matrix ℝ length p.B).symm_apply_apply _) k
  right_inv := by
    intro xs
    change (matrix ℝ length p.B) ((matrix ℝ length p.B).symm xs) = xs
    exact (matrix ℝ length p.B).apply_symm_apply _

def forwardRealView (p : ForwardLayout) (length : Nat) :
    LogicalIO (forwardIO Bool.false p length).Ctx
      ((o : Fin (forwardIO Bool.false p length).nOut) → Fin ((forwardIO Bool.false p length).oarity o) → ((forwardIO Bool.false p length).oty o).carrier) (ForwardInputs ℝ p.B length) (Fin length → Fin p.B → ℝ)
      (fun f => (forwardIO Bool.false p length).RegionalImplements (fun _ _ v => f v)) where
  inputs := forwardRealInputs p length
  outputs := forwardRealOutputs p length

def forwardComplexInputs (p : ForwardLayout) (length : Nat) :
    (forwardIO Bool.true p length).Ctx ≃ ForwardInputs ℂ p.B length where
  toFun v := ⟨(fun k => ⟨v (0 : Fin 6) k,v (1 : Fin 6) k⟩),(fun k => ⟨v (2 : Fin 6) k,v (3 : Fin 6) k⟩),matrix ℂ length p.B (fun k => ⟨v (4 : Fin 6) k,v (5 : Fin 6) k⟩)⟩
  invFun xs := Fin.cases (fun k => (xs.initial k).re) (Fin.cases (fun k => (xs.initial k).im) (Fin.cases (fun k => (xs.transition k).re) (Fin.cases (fun k => (xs.transition k).im) (Fin.cases (fun k => (((matrix ℂ length p.B).symm xs.input) k).re) (Fin.cases (fun k => (((matrix ℂ length p.B).symm xs.input) k).im) ((fun i => i.elim0)))))))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    · exact congrArg Complex.re (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
    · exact congrArg Complex.im (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
  right_inv := by
    intro xs
    apply ForwardInputs.ext
    · rfl
    · rfl
    · change (matrix ℂ length p.B) ((matrix ℂ length p.B).symm xs.input) = xs.input
      exact (matrix ℂ length p.B).apply_symm_apply _

def forwardComplexOutputs (p : ForwardLayout) (length : Nat) :
    ((o : Fin (forwardIO Bool.true p length).nOut) → Fin ((forwardIO Bool.true p length).oarity o) → ((forwardIO Bool.true p length).oty o).carrier) ≃ (Fin length → Fin p.B → ℂ) where
  toFun v := matrix ℂ length p.B (fun k => ⟨v (0 : Fin 2) k,v (1 : Fin 2) k⟩)
  invFun xs := Fin.cases (fun k => (((matrix ℂ length p.B).symm xs) k).re) (Fin.cases (fun k => (((matrix ℂ length p.B).symm xs) k).im) ((fun i => i.elim0)))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    · exact congrArg Complex.re (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
    · exact congrArg Complex.im (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
  right_inv := by
    intro xs
    change (matrix ℂ length p.B) ((matrix ℂ length p.B).symm xs) = xs
    exact (matrix ℂ length p.B).apply_symm_apply _

def forwardComplexView (p : ForwardLayout) (length : Nat) :
    LogicalIO (forwardIO Bool.true p length).Ctx
      ((o : Fin (forwardIO Bool.true p length).nOut) → Fin ((forwardIO Bool.true p length).oarity o) → ((forwardIO Bool.true p length).oty o).carrier) (ForwardInputs ℂ p.B length) (Fin length → Fin p.B → ℂ)
      (fun f => (forwardIO Bool.true p length).RegionalImplements (fun _ _ v => f v)) where
  inputs := forwardComplexInputs p length
  outputs := forwardComplexOutputs p length

def backwardRealInputs (p : BackwardLayout) (length : Nat) :
    (backwardIO Bool.false p length).Ctx ≃ BackwardInputs ℝ p.B length where
  toFun v := ⟨(v (0 : Fin 4)),(v (1 : Fin 4)),matrix ℝ (length-1) p.B (v (2 : Fin 4)),matrix ℝ length p.B (v (3 : Fin 4))⟩
  invFun xs := Fin.cases xs.initial (Fin.cases xs.transition (Fin.cases ((matrix ℝ (length-1) p.B).symm xs.saved) (Fin.cases ((matrix ℝ length p.B).symm xs.upstream) ((fun i => i.elim0)))))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    · exact congrFun ((matrix ℝ (length-1) p.B).symm_apply_apply _) k
    · exact congrFun ((matrix ℝ length p.B).symm_apply_apply _) k
  right_inv := by
    intro xs
    apply BackwardInputs.ext
    · rfl
    · rfl
    · change (matrix ℝ (length-1) p.B) ((matrix ℝ (length-1) p.B).symm xs.saved) = xs.saved
      exact (matrix ℝ (length-1) p.B).apply_symm_apply _
    · change (matrix ℝ length p.B) ((matrix ℝ length p.B).symm xs.upstream) = xs.upstream
      exact (matrix ℝ length p.B).apply_symm_apply _

def backwardRealOutputs (p : BackwardLayout) (length : Nat) :
    ((o : Fin (backwardIO Bool.false p length).nOut) → Fin ((backwardIO Bool.false p length).oarity o) → ((backwardIO Bool.false p length).oty o).carrier) ≃ BackwardOutputs ℝ p.B length where
  toFun v := ⟨matrix ℝ length p.B (v (0 : Fin 3)),(v (1 : Fin 3)),(v (2 : Fin 3))⟩
  invFun xs := Fin.cases ((matrix ℝ length p.B).symm xs.inputGradient) (Fin.cases xs.initialGradient (Fin.cases xs.transitionGradient ((fun i => i.elim0))))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    exact congrFun ((matrix ℝ length p.B).symm_apply_apply _) k
  right_inv := by
    intro xs
    apply BackwardOutputs.ext
    · change (matrix ℝ length p.B) ((matrix ℝ length p.B).symm xs.inputGradient) = xs.inputGradient
      exact (matrix ℝ length p.B).apply_symm_apply _
    · rfl
    · rfl

def backwardRealView (p : BackwardLayout) (length : Nat) :
    LogicalIO (backwardIO Bool.false p length).Ctx
      ((o : Fin (backwardIO Bool.false p length).nOut) → Fin ((backwardIO Bool.false p length).oarity o) → ((backwardIO Bool.false p length).oty o).carrier) (BackwardInputs ℝ p.B length) (BackwardOutputs ℝ p.B length)
      (fun f => (backwardIO Bool.false p length).RegionalImplements (fun _ _ v => f v)) where
  inputs := backwardRealInputs p length
  outputs := backwardRealOutputs p length

def backwardComplexInputs (p : BackwardLayout) (length : Nat) :
    (backwardIO Bool.true p length).Ctx ≃ BackwardInputs ℂ p.B length where
  toFun v := ⟨(fun k => ⟨v (0 : Fin 8) k,v (1 : Fin 8) k⟩),(fun k => ⟨v (2 : Fin 8) k,v (3 : Fin 8) k⟩),matrix ℂ (length-1) p.B (fun k => ⟨v (4 : Fin 8) k,v (5 : Fin 8) k⟩),matrix ℂ length p.B (fun k => ⟨v (6 : Fin 8) k,v (7 : Fin 8) k⟩)⟩
  invFun xs := Fin.cases (fun k => (xs.initial k).re) (Fin.cases (fun k => (xs.initial k).im) (Fin.cases (fun k => (xs.transition k).re) (Fin.cases (fun k => (xs.transition k).im) (Fin.cases (fun k => (((matrix ℂ (length-1) p.B).symm xs.saved) k).re) (Fin.cases (fun k => (((matrix ℂ (length-1) p.B).symm xs.saved) k).im) (Fin.cases (fun k => (((matrix ℂ length p.B).symm xs.upstream) k).re) (Fin.cases (fun k => (((matrix ℂ length p.B).symm xs.upstream) k).im) ((fun i => i.elim0)))))))))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    · exact congrArg Complex.re (congrFun ((matrix ℂ (length-1) p.B).symm_apply_apply _) k)
    · exact congrArg Complex.im (congrFun ((matrix ℂ (length-1) p.B).symm_apply_apply _) k)
    · exact congrArg Complex.re (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
    · exact congrArg Complex.im (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
  right_inv := by
    intro xs
    apply BackwardInputs.ext
    · rfl
    · rfl
    · change (matrix ℂ (length-1) p.B) ((matrix ℂ (length-1) p.B).symm xs.saved) = xs.saved
      exact (matrix ℂ (length-1) p.B).apply_symm_apply _
    · change (matrix ℂ length p.B) ((matrix ℂ length p.B).symm xs.upstream) = xs.upstream
      exact (matrix ℂ length p.B).apply_symm_apply _

def backwardComplexOutputs (p : BackwardLayout) (length : Nat) :
    ((o : Fin (backwardIO Bool.true p length).nOut) → Fin ((backwardIO Bool.true p length).oarity o) → ((backwardIO Bool.true p length).oty o).carrier) ≃ BackwardOutputs ℂ p.B length where
  toFun v := ⟨matrix ℂ length p.B (fun k => ⟨v (0 : Fin 6) k,v (1 : Fin 6) k⟩),(fun k => ⟨v (2 : Fin 6) k,v (3 : Fin 6) k⟩),(fun k => ⟨v (4 : Fin 6) k,v (5 : Fin 6) k⟩)⟩
  invFun xs := Fin.cases (fun k => (((matrix ℂ length p.B).symm xs.inputGradient) k).re) (Fin.cases (fun k => (((matrix ℂ length p.B).symm xs.inputGradient) k).im) (Fin.cases (fun k => (xs.initialGradient k).re) (Fin.cases (fun k => (xs.initialGradient k).im) (Fin.cases (fun k => (xs.transitionGradient k).re) (Fin.cases (fun k => (xs.transitionGradient k).im) ((fun i => i.elim0)))))))
  left_inv := by
    intro v; funext port k
    fin_cases port <;> dsimp <;> try rfl
    · exact congrArg Complex.re (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
    · exact congrArg Complex.im (congrFun ((matrix ℂ length p.B).symm_apply_apply _) k)
  right_inv := by
    intro xs
    apply BackwardOutputs.ext
    · change (matrix ℂ length p.B) ((matrix ℂ length p.B).symm xs.inputGradient) = xs.inputGradient
      exact (matrix ℂ length p.B).apply_symm_apply _
    · rfl
    · rfl

def backwardComplexView (p : BackwardLayout) (length : Nat) :
    LogicalIO (backwardIO Bool.true p length).Ctx
      ((o : Fin (backwardIO Bool.true p length).nOut) → Fin ((backwardIO Bool.true p length).oarity o) → ((backwardIO Bool.true p length).oty o).carrier) (BackwardInputs ℂ p.B length) (BackwardOutputs ℂ p.B length)
      (fun f => (backwardIO Bool.true p length).RegionalImplements (fun _ _ v => f v)) where
  inputs := backwardComplexInputs p length
  outputs := backwardComplexOutputs p length

end
end VeriTile.Bench.Repairs.DiagSsmTriton.LogicalIO

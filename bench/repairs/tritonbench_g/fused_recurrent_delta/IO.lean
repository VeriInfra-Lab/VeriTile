import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealAlgebra
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Geometry
import bench.repairs.tritonbench_g.fused_recurrent_delta.LiteralIO

/-! Actual FP32 runtime invocation of the complete ordinary quotations. The
historical scale literal is used only to select the already established
scalar-model value; actual source reads the installed scale register. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeIO
open VeriTile.Triton FP.Structural KernelProgram Interface
noncomputable section

def source (kind : Kind) (optimized : Bool) (s : Shape) (f : Flags) : ComputeKernel :=
  match kind with
  | .forward =>
    if optimized then Runtime.forwardDirectOutput "q" "k" "v" "beta" "o" "h0" "ht"
      (s.T*s.K) (s.T*s.V) s.B s.H s.T s.K s.V s.BK s.FBV f.initial f.final f.vectorBeta
    else Runtime.forward "q" "k" "v" "beta" "o" "h0" "ht"
      (s.T*s.K) (s.T*s.V) s.B s.H s.T s.K s.V s.BK s.FBV f.initial f.final f.vectorBeta
  | .backward => Runtime.backward "q" "k" "v" "beta" "dht" "dh0" "do" "dq" "dk" "dv" "dbeta" "h0"
      (s.T*s.K) (s.T*s.V) 1 s.B s.H s.T s.K s.V s.BK s.BBV
      f.initial f.vectorBeta f.initialGrad f.terminalGrad

def scaleValue {α : Type} (M : Algebra α) (scale : ℝ) : α :=
  (FP.ProfiledRewrite.engine M).literal (some .fp32) .real scale

def invocation {α : Type} (M : Algebra α) (kind : Kind) (optimized : Bool)
    (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) : Invocation α :=
  {Interface.job M kind s f scale bh iv with
    kernel := source kind optimized s f
    arguments := [⟨"scale",.real,[],fun _ => scaleValue M scale⟩]}

def literalIO {α : Type} (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) : ProgramIO α α :=
  match kind with
  | .forward => forwardIO M s f scale bh iv
  | .backward => backwardIO M s f scale bh iv

def io {α : Type} (M : Algebra α) (kind : Kind) (optimized : Bool) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) : ProgramIO α α :=
  {literalIO M kind s f scale bh iv with
    schedule := fun _ => [invocation M kind optimized s f scale bh iv]}

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeIO

/-! Independent finite-input recurrences. Masked zeros, FP32 casts, all padded
reduction lanes and the backward store/reload rounding are retained explicitly.
These functions contain no source execution or memory-state observations. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open KernelReference (zero cast add)
noncomputable section
variable {α : Type} [Inhabited α]

def forwardData (M : Algebra α) (s : Shape) (f : Flags) (iv : Nat)
    (vals : ForwardInputs α s) : KernelReference.ForwardInput s α where
  query := fun t i => if ht : t<s.T then KernelReference.cast M (if i.val<s.K then vals (0 : Fin 6) (⟨t,ht⟩,i,()) else zero M) else zero M
  key := fun t i => if ht : t<s.T then KernelReference.cast M (if i.val<s.K then vals (1 : Fin 6) (⟨t,ht⟩,i,()) else zero M) else zero M
  value := fun t j => if ht : t<s.T then KernelReference.cast M (if iv*s.FBV+j.val<s.V then vals (2 : Fin 6) (⟨t,ht⟩,j,()) else zero M) else zero M
  beta := fun t j => if ht : t<s.T then
      if f.vectorBeta then KernelReference.cast M (if iv*s.FBV+j.val<s.V then vals (3 : Fin 6) (⟨t,ht⟩,j,()) else zero M)
      else KernelReference.cast M (vals (4 : Fin 6) (⟨t,ht⟩,())) else zero M
  seed := fun ji => if f.initial then add M (zero M)
    (KernelReference.cast M (if ji.2.1.val<s.K ∧ iv*s.FBV+ji.1.val<s.V then vals (5 : Fin 6) (ji.2.1,ji.1,()) else zero M)) else zero M

def forwardReference (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (iv : Nat)
    (vals : ForwardInputs α s) : ForwardOutputs α s
  | ⟨0,_⟩ => fun j => KernelReference.cast M (KernelReference.output M s (forwardData M s f iv vals) scale j.1.val j.2.1)
  | ⟨1,_⟩ => fun j => KernelReference.cast M (KernelReference.residual M s (forwardData M s f iv vals) j.1.val j.2.1)
  | ⟨2,_⟩ => fun j => KernelReference.cast M (KernelReference.forward M s (forwardData M s f iv vals) s.T (j.2.1,j.1,()))
  | ⟨n+3,h⟩ => False.elim (by simp [forwardOutputs] at h)

def backwardData (M : Algebra α) (s : Shape) (f : Flags) (iv : Nat)
    (vals : BackwardInputs α s) : KernelBackwardReference.Input s α where
  query := fun t i => if ht : t<s.T then KernelReference.cast M (if i.val<s.K then vals (0 : Fin 8) (⟨t,ht⟩,i,()) else zero M) else zero M
  key := fun t i => if ht : t<s.T then KernelReference.cast M (if i.val<s.K then vals (1 : Fin 8) (⟨t,ht⟩,i,()) else zero M) else zero M
  value := fun t j => if ht : t<s.T then KernelReference.cast M (if iv*s.BBV+j.val<s.V then vals (2 : Fin 8) (⟨t,ht⟩,j,()) else zero M) else zero M
  beta := fun t j => if ht : t<s.T then
      if f.vectorBeta then KernelReference.cast M (if iv*s.BBV+j.val<s.V then vals (3 : Fin 8) (⟨t,ht⟩,j,()) else zero M)
      else KernelReference.cast M (vals (4 : Fin 8) (⟨t,ht⟩,())) else zero M
  incoming := fun t j => if ht : t<s.T then KernelReference.cast M (if iv*s.BBV+j.val<s.V then vals (5 : Fin 8) (⟨t,ht⟩,j,()) else zero M) else zero M
  initial := fun ij => if f.initial then add M (zero M)
    (KernelReference.cast M (if ij.1.val<s.K ∧ iv*s.BBV+ij.2.1.val<s.V then vals (6 : Fin 8) ij else zero M)) else zero M
  terminal := fun ij => if f.terminalGrad then add M (zero M)
    (KernelReference.cast M (if ij.1.val<s.K ∧ iv*s.BBV+ij.2.1.val<s.V then vals (7 : Fin 8) ij else zero M)) else zero M

def enhanced (M : Algebra α) (s : Shape) (scale : ℝ)
    (a : KernelBackwardReference.Input s α) (t : Nat) :=
  KernelBackwardReference.enhance M s scale a t (KernelBackwardReference.reverse M s scale a (s.T-1-t))

def backwardQuery (M : Algebra α) (s : Shape) (scale : ℝ)
    (a : KernelBackwardReference.Input s α) (t : Nat) :=
  KernelBackwardReference.dq M s scale a t (KernelBackwardReference.forward M s a (t+1))

def backwardKey (M : Algebra α) (s : Shape) (scale : ℝ) (iv : Nat)
    (a : KernelBackwardReference.Input s α) (t : Nat) :=
  KernelBackwardReference.correction M s (KernelBackwardReference.forward M s a t)
    (fun i => KernelReference.cast M (if i.val<s.K then KernelReference.cast M
      (KernelBackwardReference.dk M s a t (enhanced M s scale a t) i) else zero M))
    (fun j => KernelReference.cast M (if iv*s.BBV+j.val<s.V then KernelReference.cast M
      (KernelBackwardReference.dv M s a t (enhanced M s scale a t) j) else zero M))

def backwardReference (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (iv : Nat)
    (vals : BackwardInputs α s) : BackwardOutputs α s
  | ⟨0,_⟩ => fun j => KernelReference.cast M (backwardQuery M s scale (backwardData M s f iv vals) j.1.val j.2.1)
  | ⟨1,_⟩ => fun j => KernelReference.cast M (backwardKey M s scale iv (backwardData M s f iv vals) j.1.val j.2.1)
  | ⟨2,_⟩ => fun j => KernelReference.cast M (KernelBackwardReference.dv M s (backwardData M s f iv vals) j.1.val
      (enhanced M s scale (backwardData M s f iv vals) j.1.val) j.2.1)
  | ⟨3,_⟩ => fun j => KernelReference.cast M (KernelBackwardReference.dbeta M s (backwardData M s f iv vals) j.1.val
      (enhanced M s scale (backwardData M s f iv vals) j.1.val) j.2.1)
  | ⟨4,_⟩ => fun j => KernelReference.cast M (KernelBackwardReference.betaSum M s
      (KernelBackwardReference.dbeta M s (backwardData M s f iv vals) j.1.val
        (enhanced M s scale (backwardData M s f iv vals) j.1.val)))
  | ⟨5,_⟩ => fun j => KernelReference.cast M (KernelBackwardReference.reverse M s scale (backwardData M s f iv vals) s.T j)
  | ⟨n+6,h⟩ => False.elim (by simp [backwardOutputs] at h)

#specNonCircular forwardReference avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source,KernelReference.forwardInput]
#specNonCircular backwardReference avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source,KernelBackwardReference.input]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section

def wideKey (s : Shape) (i : Fin s.K) : Fin s.BK :=
  ⟨i.val,lt_of_lt_of_le i.isLt (Geometry.nextPower_ge s.K)⟩

def primal (s : Shape) (f : Flags) (vals : ForwardInputs ℝ s) (j : Fin s.FBV) : Math.Input s.K where
  q := fun t i => if ht : t<s.T then vals (0 : Fin 6) (⟨t,ht⟩,wideKey s i,()) else 0
  k := fun t i => if ht : t<s.T then vals (1 : Fin 6) (⟨t,ht⟩,wideKey s i,()) else 0
  x := fun t => if ht : t<s.T then vals (2 : Fin 6) (⟨t,ht⟩,j,()) else 0
  beta := fun t => if ht : t<s.T then
    if f.vectorBeta then vals (3 : Fin 6) (⟨t,ht⟩,j,()) else vals (4 : Fin 6) (⟨t,ht⟩,()) else 0
  h0 := fun i => if f.initial then vals (5 : Fin 6) (wideKey s i,j,()) else 0

structure SavedInputs (s : Shape) (f : Flags) (iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ)
    (terminal : Fin s.BBV → Math.Vec s.BK) (vals : BackwardInputs ℝ s) : Prop where
  query : ∀ t,t<s.T → ∀ i j,(backwardData KernelReal.algebra s f iv vals).query t i=(a j).q t i
  key : ∀ t,t<s.T → ∀ i j,(backwardData KernelReal.algebra s f iv vals).key t i=(a j).k t i
  beta : ∀ t,t<s.T → ∀ j,(backwardData KernelReal.algebra s f iv vals).beta t j=(a j).beta t
  residual : ∀ t,t<s.T → ∀ j,(backwardData KernelReal.algebra s f iv vals).value t j=
    Math.residual (Math.state (a j) t) ((a j).k t) ((a j).x t)
  incoming : ∀ t,t<s.T → ∀ j,(backwardData KernelReal.algebra s f iv vals).incoming t j=g t j
  initial : ∀ i j,(backwardData KernelReal.algebra s f iv vals).initial (i,j,())=(a j).h0 i
  terminal : ∀ i j,(backwardData KernelReal.algebra s f iv vals).terminal (i,j,())=terminal j i

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
open VeriTile.Triton FP.Structural
noncomputable section
structure ForwardTensorInputs (α : Type) (c : Shape) where
  query : Fin (c.T) → Fin (c.BK) → α
  key : Fin (c.T) → Fin (c.BK) → α
  value : Fin (c.T) → Fin (c.FBV) → α
  vectorBeta : Fin (c.T) → Fin (c.FBV) → α
  scalarBeta : Fin (c.T) → α
  initialState : Fin (c.BK) → Fin (c.FBV) → α

def forwardInputCoordinates (α : Type) (c : Shape) : ChannelValues α (forwardInputs c) ≃ ForwardTensorInputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.T, c.BK] (xs (0 : Fin 6)),
    Values.curryEquiv α .real [c.T, c.BK] (xs (1 : Fin 6)),
    Values.curryEquiv α .real [c.T, c.FBV] (xs (2 : Fin 6)),
    Values.curryEquiv α .real [c.T, c.FBV] (xs (3 : Fin 6)),
    Values.curryEquiv α .real [c.T] (xs (4 : Fin 6)),
    Values.curryEquiv α .real [c.BK, c.FBV] (xs (5 : Fin 6))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.T, c.BK]).symm xs.query) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BK]).symm xs.key) (Fin.cases ((Values.curryEquiv α .real [c.T, c.FBV]).symm xs.value) (Fin.cases ((Values.curryEquiv α .real [c.T, c.FBV]).symm xs.vectorBeta) (Fin.cases ((Values.curryEquiv α .real [c.T]).symm xs.scalarBeta) (Fin.cases ((Values.curryEquiv α .real [c.BK, c.FBV]).symm xs.initialState) (fun i => Fin.elim0 i)))))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.T, c.BK]).symm_apply_apply (xs (0 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T, c.BK]).symm_apply_apply (xs (1 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T, c.FBV]).symm_apply_apply (xs (2 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T, c.FBV]).symm_apply_apply (xs (3 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T]).symm_apply_apply (xs (4 : Fin 6))
    · exact (Values.curryEquiv α .real [c.BK, c.FBV]).symm_apply_apply (xs (5 : Fin 6))
  right_inv := by intro xs; cases xs; rfl

structure ForwardTensorOutputs (α : Type) (c : Shape) where
  output : Fin (c.T) → Fin (c.FBV) → α
  residual : Fin (c.T) → Fin (c.FBV) → α
  finalState : Fin (c.BK) → Fin (c.FBV) → α

def forwardOutputCoordinates (α : Type) (c : Shape) : ChannelValues α (forwardOutputs c) ≃ ForwardTensorOutputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.T, c.FBV] (xs (0 : Fin 3)),
    Values.curryEquiv α .real [c.T, c.FBV] (xs (1 : Fin 3)),
    Values.curryEquiv α .real [c.BK, c.FBV] (xs (2 : Fin 3))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.T, c.FBV]).symm xs.output) (Fin.cases ((Values.curryEquiv α .real [c.T, c.FBV]).symm xs.residual) (Fin.cases ((Values.curryEquiv α .real [c.BK, c.FBV]).symm xs.finalState) (fun i => Fin.elim0 i))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.T, c.FBV]).symm_apply_apply (xs (0 : Fin 3))
    · exact (Values.curryEquiv α .real [c.T, c.FBV]).symm_apply_apply (xs (1 : Fin 3))
    · exact (Values.curryEquiv α .real [c.BK, c.FBV]).symm_apply_apply (xs (2 : Fin 3))
  right_inv := by intro xs; cases xs; rfl

structure BackwardTensorInputs (α : Type) (c : Shape) where
  query : Fin (c.T) → Fin (c.BK) → α
  key : Fin (c.T) → Fin (c.BK) → α
  residual : Fin (c.T) → Fin (c.BBV) → α
  vectorBeta : Fin (c.T) → Fin (c.BBV) → α
  scalarBeta : Fin (c.T) → α
  incoming : Fin (c.T) → Fin (c.BBV) → α
  initialState : Fin (c.BK) → Fin (c.BBV) → α
  terminalGradient : Fin (c.BK) → Fin (c.BBV) → α

def backwardInputCoordinates (α : Type) (c : Shape) : ChannelValues α (backwardInputs c) ≃ BackwardTensorInputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.T, c.BK] (xs (0 : Fin 8)),
    Values.curryEquiv α .real [c.T, c.BK] (xs (1 : Fin 8)),
    Values.curryEquiv α .real [c.T, c.BBV] (xs (2 : Fin 8)),
    Values.curryEquiv α .real [c.T, c.BBV] (xs (3 : Fin 8)),
    Values.curryEquiv α .real [c.T] (xs (4 : Fin 8)),
    Values.curryEquiv α .real [c.T, c.BBV] (xs (5 : Fin 8)),
    Values.curryEquiv α .real [c.BK, c.BBV] (xs (6 : Fin 8)),
    Values.curryEquiv α .real [c.BK, c.BBV] (xs (7 : Fin 8))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.T, c.BK]).symm xs.query) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BK]).symm xs.key) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BBV]).symm xs.residual) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BBV]).symm xs.vectorBeta) (Fin.cases ((Values.curryEquiv α .real [c.T]).symm xs.scalarBeta) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BBV]).symm xs.incoming) (Fin.cases ((Values.curryEquiv α .real [c.BK, c.BBV]).symm xs.initialState) (Fin.cases ((Values.curryEquiv α .real [c.BK, c.BBV]).symm xs.terminalGradient) (fun i => Fin.elim0 i)))))))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.T, c.BK]).symm_apply_apply (xs (0 : Fin 8))
    · exact (Values.curryEquiv α .real [c.T, c.BK]).symm_apply_apply (xs (1 : Fin 8))
    · exact (Values.curryEquiv α .real [c.T, c.BBV]).symm_apply_apply (xs (2 : Fin 8))
    · exact (Values.curryEquiv α .real [c.T, c.BBV]).symm_apply_apply (xs (3 : Fin 8))
    · exact (Values.curryEquiv α .real [c.T]).symm_apply_apply (xs (4 : Fin 8))
    · exact (Values.curryEquiv α .real [c.T, c.BBV]).symm_apply_apply (xs (5 : Fin 8))
    · exact (Values.curryEquiv α .real [c.BK, c.BBV]).symm_apply_apply (xs (6 : Fin 8))
    · exact (Values.curryEquiv α .real [c.BK, c.BBV]).symm_apply_apply (xs (7 : Fin 8))
  right_inv := by intro xs; cases xs; rfl

structure BackwardTensorOutputs (α : Type) (c : Shape) where
  dq : Fin (c.T) → Fin (c.BK) → α
  dk : Fin (c.T) → Fin (c.BK) → α
  dv : Fin (c.T) → Fin (c.BBV) → α
  vectorDBeta : Fin (c.T) → Fin (c.BBV) → α
  scalarDBeta : Fin (c.T) → α
  dh0 : Fin (c.BK) → Fin (c.BBV) → α

def backwardOutputCoordinates (α : Type) (c : Shape) : ChannelValues α (backwardOutputs c) ≃ BackwardTensorOutputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.T, c.BK] (xs (0 : Fin 6)),
    Values.curryEquiv α .real [c.T, c.BK] (xs (1 : Fin 6)),
    Values.curryEquiv α .real [c.T, c.BBV] (xs (2 : Fin 6)),
    Values.curryEquiv α .real [c.T, c.BBV] (xs (3 : Fin 6)),
    Values.curryEquiv α .real [c.T] (xs (4 : Fin 6)),
    Values.curryEquiv α .real [c.BK, c.BBV] (xs (5 : Fin 6))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.T, c.BK]).symm xs.dq) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BK]).symm xs.dk) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BBV]).symm xs.dv) (Fin.cases ((Values.curryEquiv α .real [c.T, c.BBV]).symm xs.vectorDBeta) (Fin.cases ((Values.curryEquiv α .real [c.T]).symm xs.scalarDBeta) (Fin.cases ((Values.curryEquiv α .real [c.BK, c.BBV]).symm xs.dh0) (fun i => Fin.elim0 i)))))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.T, c.BK]).symm_apply_apply (xs (0 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T, c.BK]).symm_apply_apply (xs (1 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T, c.BBV]).symm_apply_apply (xs (2 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T, c.BBV]).symm_apply_apply (xs (3 : Fin 6))
    · exact (Values.curryEquiv α .real [c.T]).symm_apply_apply (xs (4 : Fin 6))
    · exact (Values.curryEquiv α .real [c.BK, c.BBV]).symm_apply_apply (xs (5 : Fin 6))
  right_inv := by intro xs; cases xs; rfl

def forwardPrimal (s : Shape) (f : Flags) (xs : ForwardTensorInputs ℝ s) :=
  primal s f ((forwardInputCoordinates ℝ s).symm xs)

def savedDomain (s : Shape) (f : Flags) (iv : Nat) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) (xs : BackwardTensorInputs ℝ s) : Prop :=
  SavedInputs s f iv a g terminal ((backwardInputCoordinates ℝ s).symm xs)

def forwardLogicalIO (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    LogicalProgramIO ℝ ℝ (ForwardTensorInputs ℝ s) (ForwardTensorOutputs ℝ s) where
  core := RuntimeIO.io KernelReal.algebra .forward Bool.false s f scale bh iv
  inputs := forwardInputCoordinates ℝ s
  outputs := forwardOutputCoordinates ℝ s

def backwardLogicalIO (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    LogicalProgramIO ℝ ℝ (BackwardTensorInputs ℝ s) (BackwardTensorOutputs ℝ s) where
  core := RuntimeIO.io KernelReal.algebra .backward Bool.false s f scale bh iv
  inputs := backwardInputCoordinates ℝ s
  outputs := backwardOutputCoordinates ℝ s
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface

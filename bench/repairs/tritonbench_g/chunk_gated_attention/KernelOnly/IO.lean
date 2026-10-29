import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.DirectRealWhole
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.DirectCumulativeWhole
import VeriTile.Triton.Memory.AlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

/-! Finite input/output interfaces for the two selected FP32 kernels. Real
interpretation uses the existing dtype-erased source projection. Each plan is
an ordinary merged grid, and plans execute with successive memories. Logical
outputs and padded working cells are distinguished explicitly. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section

/-- Embed the already source-derived Algorithm projection without maintaining
a second handwritten body. -/
def selected (kernel : ComputeKernel) : AlgorithmSource :=
  let k := kernel.eraseDType.toAlgKernel
  ⟨ComputeKernel.fromKernelBody k.inputs k.outputs k.body,
    by simp [ComputeKernel.toAlgKernel]⟩

@[simp] theorem selected_projection (kernel : ComputeKernel) :
    (selected kernel).kernel.toAlgKernel = kernel.eraseDType.toAlgKernel := by
  simp [selected,ComputeKernel.toAlgKernel]

def innerStages (p : RealHMath.Layout) (r : RealHProgram.Regions)
    (gateK hasSeed storeFinal : Bool) (plans : List LaunchPlan.Item) : List AlgorithmStage :=
  plans.map fun plan => .ordinary
    (selected (RebasedProgram.source p r .fp32 .fp32 .fp32 .fp32 .fp32
      plan.base gateK hasSeed storeFinal Bool.false)) (Batch.grid plan.size)

def prefixStages (p : RealCumProgram.Layout) (heads : Nat) (input output : RegionName)
    (plans : List LaunchPlan.Item) : List AlgorithmStage :=
  plans.map fun plan => .ordinary
    (selected (RebasedCumProgram.source p input output heads plan.base Bool.false .fp32 .fp32))
    (Batch.grid plan.size)

abbrev PrefixCoordinate (heads T S : Nat) := Fin heads × Fin T × Fin S
def prefixCoordinates (heads T S : Nat) := Fintype.equivFin (PrefixCoordinate heads T S)
def prefixCount (heads T S : Nat) := Fintype.card (PrefixCoordinate heads T S)

def prefixIO (p : RealCumProgram.Layout) (heads T S : Nat)
    (input output : RegionName) : AlgorithmProgramIO where
  nIn := 1
  nOut := 1
  nScr := 1
  bufs := [input,output]
  ity := fun _ => .float
  iarity := fun _ => heads*p.T*p.S
  ibuf := fun _ => input
  oty := fun _ => .float
  oarity := fun _ => prefixCount heads T S
  obuf := fun _ => output
  obuf_mem := fun _ => by simp
  sarity := fun _ => heads*p.T*p.S
  sbuf := fun _ => output
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j =>
    let c := (prefixCoordinates heads T S).symm j
    c.1.val*p.headStride+c.2.1.val*p.timeStride+c.2.2.val*p.channelStride
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True
  initialUndef := .zero
  stages := fun _ => prefixStages p heads input output (DirectCumulativeWhole.plans p heads)

/-- Ordinary input staging: zero padding outside the logical dimensions.
Only the finite declared input window is constrained. -/
def prefixDomain (p : RealCumProgram.Layout) (heads T S : Nat)
    (x : Nat → Nat → Nat → ℝ) (values : Fin 1 → Fin (heads*p.T*p.S) → ℝ) : Prop :=
  ∀ j, values 0 j = CumulativeHost.materialized p T S x j.val

/-- Independent within-chunk cumulative sum at every logical output. -/
def prefixFormula (p : RealCumProgram.Layout) (heads T S : Nat)
    (x : Nat → Nat → Nat → ℝ) (_ : Fin 1 → Fin (heads*p.T*p.S) → ℝ)
    (_ : Fin 1) (j : Fin (prefixCount heads T S)) : ℝ :=
  let c := (prefixCoordinates heads T S).symm j
  Host.chunkPrefix p.BT (x c.1.val) c.2.1.val c.2.2.val

def innerInputArity (p : RealHMath.Layout) (heads : Nat) (gateK : Bool) (i : Fin 4) : Nat :=
  match i with
  | ⟨0,_⟩ => heads*p.T*p.K
  | ⟨1,_⟩ => heads*p.T*p.V
  | ⟨2,_⟩ => heads*p.T*(if gateK then p.K else p.V)
  | ⟨3,_⟩ => heads*p.K*p.V

def innerInputRegion (r : RealHProgram.Regions) (i : Fin 4) : RegionName :=
  match i with
  | ⟨0,_⟩ => r.key | ⟨1,_⟩ => r.value | ⟨2,_⟩ => r.gate | ⟨3,_⟩ => r.seed

def innerInputActive (hasSeed : Bool) (i : Fin 4) : Prop := i.val ≠ 3 ∨ hasSeed = Bool.true

abbrev HistoryCoordinate (heads NT K V : Nat) := Fin heads × Fin NT × Fin K × Fin V
abbrev FinalCoordinate (heads K V : Nat) := Fin heads × Fin K × Fin V
def historyCoordinates (heads NT K V : Nat) := Fintype.equivFin (HistoryCoordinate heads NT K V)
def finalCoordinates (heads K V : Nat) := Fintype.equivFin (FinalCoordinate heads K V)
def innerOutputArity (p : RealHMath.Layout) (heads K V : Nat) (o : Fin 2) : Nat :=
  if o.val=0 then Fintype.card (HistoryCoordinate heads p.NT K V)
  else Fintype.card (FinalCoordinate heads K V)
def innerScratchArity (p : RealHMath.Layout) (heads : Nat) (o : Fin 2) : Nat :=
  if o.val=0 then heads*p.NT*p.K*p.V else heads*p.K*p.V
def innerOutputRegion (r : RealHProgram.Regions) (o : Fin 2) : RegionName :=
  if o.val=0 then r.history else r.final
def innerOutputActive (storeFinal : Bool) (o : Fin 2) : Prop := o.val=0 ∨ storeFinal=Bool.true

def innerOutputOffset (p : RealHMath.Layout) (heads K V : Nat) (o : Fin 2) :
    Fin (innerOutputArity p heads K V o) → Nat :=
  if h : o.val=0 then fun j =>
    let c := (historyCoordinates heads p.NT K V).symm (Fin.cast (by simp [innerOutputArity,h]) j)
    DirectRealWhole.historyOffset p c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val
  else fun j =>
    let c := (finalCoordinates heads K V).symm (Fin.cast (by simp [innerOutputArity,h]) j)
    DirectRealWhole.finalOffset p c.1.val c.2.1.val c.2.2.val

def innerIO (p : RealHMath.Layout) (heads K V : Nat) (r : RealHProgram.Regions)
    (gateK hasSeed storeFinal : Bool) : AlgorithmProgramIO where
  nIn := 4
  nOut := 2
  nScr := 2
  bufs := [r.key,r.value,r.gate,r.seed,r.history,r.final]
  ity := fun _ => .float
  iarity := innerInputArity p heads gateK
  ibuf := innerInputRegion r
  oty := fun _ => .float
  oarity := innerOutputArity p heads K V
  obuf := innerOutputRegion r
  obuf_mem := fun o => by by_cases h : o.val=0 <;> simp [innerOutputRegion,h]
  sarity := innerScratchArity p heads
  sbuf := innerOutputRegion r
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun i _ _ _ _ _ => innerInputActive hasSeed i
  owin := fun o _ _ _ _ j => innerOutputOffset p heads K V o j
  omask := fun o _ _ _ _ _ => innerOutputActive storeFinal o
  swin := fun _ _ _ _ _ j => j.val
  smask := fun o _ _ _ _ _ => innerOutputActive storeFinal o
  initialUndef := .zero
  stages := fun _ => innerStages p r gateK hasSeed storeFinal
    (LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV)))

def innerInputValues (a : RealHMath.Inputs) (i : Fin 4) : Nat → ℝ :=
  match i with
  | ⟨0,_⟩ => a.key | ⟨1,_⟩ => a.value | ⟨2,_⟩ => a.gate | ⟨3,_⟩ => a.seed

/-- Explicit finite zero-padding/repeated-tail contract. Absent seed is not
read and its channel is inactive. Input roles may alias one another. -/
def innerDomain (p : RealHMath.Layout) (heads T K V : Nat) (gateK hasSeed : Bool)
    (x : Nat → Host.InnerInputs)
    (values : (i : Fin 4) → Fin (innerInputArity p heads gateK i) → ℝ) : Prop :=
  ∀ i j, innerInputActive hasSeed i →
    values i j = innerInputValues (Materialize.inputs p T K V gateK x) i j.val

def innerFormula (p : RealHMath.Layout) (heads T K V : Nat) (gateK hasSeed : Bool)
    (x : Nat → Host.InnerInputs)
    (_ : (i : Fin 4) → Fin (innerInputArity p heads gateK i) → ℝ)
    (o : Fin 2) : Fin (innerOutputArity p heads K V o) → ℝ :=
  if h : o.val=0 then fun j =>
    let c := (historyCoordinates heads p.NT K V).symm (Fin.cast (by simp [innerOutputArity,h]) j)
    Host.innerState (x c.1.val) T p.BT gateK hasSeed c.2.1.val c.2.2.1.val c.2.2.2.val
  else fun j =>
    let c := (finalCoordinates heads K V).symm (Fin.cast (by simp [innerOutputArity,h]) j)
    Host.innerState (x c.1.val) T p.BT gateK hasSeed p.NT c.2.1.val c.2.2.val

#specNonCircular prefixFormula avoiding [exec,execR,AlgorithmExecution]
#specNonCircular innerFormula avoiding [exec,execR,AlgorithmExecution]
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface.Logical
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section

def coords3 (a b c : Nat) : (Fin a × Fin b × Fin c) ≃ Fin (a*b*c) :=
  (Equiv.prodAssoc _ _ _).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv)
def tensor3 (a b c : Nat) : (Fin (a*b*c) → ℝ) ≃ (Fin a → Fin b → Fin c → ℝ) where
  toFun := fun f i j k => f (coords3 a b c (i,j,k))
  invFun := fun f i => f ((coords3 a b c).symm i).1 ((coords3 a b c).symm i).2.1 ((coords3 a b c).symm i).2.2
  left_inv := by
    intro f
    funext i
    exact congrArg f ((coords3 a b c).apply_symm_apply i)
  right_inv := by
    intro f
    funext i j k
    change f ((coords3 a b c).symm (coords3 a b c (i,j,k))).1
      ((coords3 a b c).symm (coords3 a b c (i,j,k))).2.1
      ((coords3 a b c).symm (coords3 a b c (i,j,k))).2.2 = f i j k
    rw [Equiv.symm_apply_apply]

/-- All padded input tensor cells are retained, even when the domain makes a
source role inactive. Logical x remains the explicit independent staged input. -/
structure InnerInputs (p : RealHMath.Layout) (heads : Nat) (gateK : Bool) where
  key : Fin heads → Fin p.T → Fin p.K → ℝ
  value : Fin heads → Fin p.T → Fin p.V → ℝ
  gate : Fin heads → Fin p.T → Fin (if gateK then p.K else p.V) → ℝ
  seed : Fin heads → Fin p.K → Fin p.V → ℝ

structure InnerOutputs (p : RealHMath.Layout) (heads K V : Nat) where
  history : Fin heads → Fin p.NT → Fin K → Fin V → ℝ
  final : Fin heads → Fin K → Fin V → ℝ

def innerInputs (p : RealHMath.Layout) (heads : Nat) (gateK : Bool) :
    ((i : Fin 4) → Fin (innerInputArity p heads gateK i) → ℝ) ≃ InnerInputs p heads gateK where
  toFun := fun values => ⟨tensor3 heads p.T p.K (values 0),
    tensor3 heads p.T p.V (values 1),
    tensor3 heads p.T (if gateK then p.K else p.V) (values 2),
    tensor3 heads p.K p.V (values 3)⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (tensor3 heads p.T p.K).symm values.key
    | ⟨1,_⟩ => (tensor3 heads p.T p.V).symm values.value
    | ⟨2,_⟩ => (tensor3 heads p.T (if gateK then p.K else p.V)).symm values.gate
    | ⟨3,_⟩ => (tensor3 heads p.K p.V).symm values.seed
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def innerOutputs (p : RealHMath.Layout) (heads K V : Nat) :
    ((o : Fin 2) → Fin (innerOutputArity p heads K V o) → ℝ) ≃ InnerOutputs p heads K V where
  toFun := fun values => ⟨
    fun h c k v => values 0 (historyCoordinates heads p.NT K V (h,c,k,v)),
    fun h k v => values 1 (finalCoordinates heads K V (h,k,v))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => fun j => let c := (historyCoordinates heads p.NT K V).symm j
      values.history c.1 c.2.1 c.2.2.1 c.2.2.2
    | ⟨1,_⟩ => fun j => let c := (finalCoordinates heads K V).symm j
      values.final c.1 c.2.1 c.2.2
  left_inv := by
    intro values
    funext channel j
    fin_cases channel
    · exact congrArg (values 0) ((historyCoordinates heads p.NT K V).apply_symm_apply j)
    · exact congrArg (values 1) ((finalCoordinates heads K V).apply_symm_apply j)
  right_inv := by
    intro values
    cases values with
    | mk history final =>
      apply congrArg₂ (@InnerOutputs.mk p heads K V)
      · funext h c k v
        exact congrArg (fun c => history c.1 c.2.1 c.2.2.1 c.2.2.2)
          ((historyCoordinates heads p.NT K V).symm_apply_apply (h,c,k,v))
      · funext h k v
        exact congrArg (fun c => final c.1 c.2.1 c.2.2)
          ((finalCoordinates heads K V).symm_apply_apply (h,k,v))

def innerDomain (p : RealHMath.Layout) (heads T K V : Nat) (gateK hasSeed : Bool)
    (x : Nat → Host.InnerInputs) (values : InnerInputs p heads gateK) : Prop :=
  RealInterface.innerDomain p heads T K V gateK hasSeed x ((innerInputs p heads gateK).symm values)

def innerIO (p : RealHMath.Layout) (heads K V : Nat) (r : RealHProgram.Regions)
    (gateK hasSeed storeFinal : Bool) :
    LogicalDomainIO ((RealInterface.innerIO p heads K V r gateK hasSeed storeFinal).Ctx)
      ((o : Fin 2) → Fin (innerOutputArity p heads K V o) → ℝ)
      (InnerInputs p heads gateK) (InnerOutputs p heads K V)
      (RealInterface.innerIO p heads K V r gateK hasSeed storeFinal).ImplementsOn where
  inputs := innerInputs p heads gateK
  outputs := innerOutputs p heads K V

def prefixInputs (p : RealCumProgram.Layout) (heads : Nat) :
    (Fin 1 → Fin (heads*p.T*p.S) → ℝ) ≃ (Fin heads → Fin p.T → Fin p.S → ℝ) where
  toFun := fun values => tensor3 heads p.T p.S (values 0)
  invFun := fun values _ => (tensor3 heads p.T p.S).symm values
  left_inv := by
    intro values
    funext channel j
    have channel_zero : channel = 0 := Fin.eq_zero channel
    subst channel
    exact congrFun ((tensor3 heads p.T p.S).symm_apply_apply (values 0)) j
  right_inv := fun values => (tensor3 heads p.T p.S).apply_symm_apply values

def prefixOutputs (heads T S : Nat) :
    (Fin 1 → Fin (prefixCount heads T S) → ℝ) ≃ (Fin heads → Fin T → Fin S → ℝ) where
  toFun := fun values h t s => values 0 (prefixCoordinates heads T S (h,t,s))
  invFun := fun values _ j => let c := (prefixCoordinates heads T S).symm j
    values c.1 c.2.1 c.2.2
  left_inv := by
    intro values
    funext channel j
    have channel_zero : channel = 0 := Fin.eq_zero channel
    subst channel
    exact congrArg (values 0) ((prefixCoordinates heads T S).apply_symm_apply j)
  right_inv := by
    intro values
    funext h t s
    exact congrArg (fun c => values c.1 c.2.1 c.2.2)
      ((prefixCoordinates heads T S).symm_apply_apply (h,t,s))

def prefixDomain (p : RealCumProgram.Layout) (heads T S : Nat)
    (x : Nat → Nat → Nat → ℝ) (values : Fin heads → Fin p.T → Fin p.S → ℝ) : Prop :=
  RealInterface.prefixDomain p heads T S x ((prefixInputs p heads).symm values)

def prefixIO (p : RealCumProgram.Layout) (heads T S : Nat) (input output : RegionName) :
    LogicalDomainIO ((RealInterface.prefixIO p heads T S input output).Ctx)
      (Fin 1 → Fin (prefixCount heads T S) → ℝ)
      (Fin heads → Fin p.T → Fin p.S → ℝ) (Fin heads → Fin T → Fin S → ℝ)
      (RealInterface.prefixIO p heads T S input output).ImplementsOn where
  inputs := prefixInputs p heads
  outputs := prefixOutputs heads T S
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface.Logical

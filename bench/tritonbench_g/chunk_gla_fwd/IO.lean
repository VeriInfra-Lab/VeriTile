import VeriTile.Triton.Float.FillProgramIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.KernelProfile
import VeriTile.Triton.Memory.AlgorithmFillProgramIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.KernelReal
import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeLaunch
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open FPStages (Stage)
noncomputable section
variable {α : Type} [Inhabited α]
def job (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) : Invocation α where
  kernel := stage.kernel
  backend := engine (stage.algebra M D)
  clearRegisters := Bool.true
  pids := idx.toPids
  numPids := some stage.grid.toNumPids

def start (stage : Stage) (idx : GridIndex stage.grid) (s : State α) : State α :=
  {s with regs:=fun _ _ _=>none,pids:=idx.toPids,numPids:=stage.grid.toNumPids}

def inputExtents : Stage → List (RegionName × Nat)
  | .inter c _ | .intra c _ _ | .split c _ =>
    [("q",c.heads*c.T*c.K),("k",c.heads*c.T*c.K),("g",c.heads*c.T*c.K)]
  | .merge c => [("scratch",Geometry.ceilDiv c.K c.BK*c.heads*c.T*16)]
  | .output c _ =>
    [("q",c.heads*c.T*c.K),("g",c.heads*c.T*c.K),("v",c.heads*c.T*c.V),
     ("h",c.heads*(Geometry.ceilDiv c.T c.BT*c.K)*c.V),("attention",c.heads*c.T*c.BT)]

def Bounds (stage : Stage) (bounds : RegionName → Nat) : Prop :=
  (∀ port∈inputExtents stage,port.2≤bounds port.1) ∧ stage.size≤bounds stage.region

def jobs (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage) : List (FillStage α) :=
  (FPGrid.jobs stage.grid).map (fun idx=>.source (job M D stage idx))

def zero (M : Algebra α) (region : RegionName) (count : Nat) : FiniteFill α :=
  ⟨region,0,count,.mk .real (M.literal (some .fp32) .real 0)⟩

def producer (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions)
    (scale : ℝ) : List (FillStage α) :=
  .fill (zero M "attention" (Host.attentionSize c)) ::
  if c.heads*c.T=0 then [] else
    jobs M D (.inter c.producer scale) ++
    if c.K≤256 then jobs M D (.intra c.producer c.intraBK scale)
    else .fill (zero M "scratch" (Host.scratchSize c)) ::
      (jobs M D (.split c.split scale) ++ jobs M D (.merge c.split))

def consumer (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions)
    (scale : ℝ) : List (FillStage α) :=
  if Host.outputSize c=0 then [] else jobs M D (.output c.consumer scale)

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeLaunch


namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativePorts
open VeriTile Triton FP.Structural NativeInstructions
noncomputable section

abbrev Word := BitVec 32
abbrev Expr := NativeStaging.Expr

def producerInputs (c : Host.Dimensions) : List IOChannel :=
  [⟨.real,[c.heads*c.T*c.K],"q"⟩,⟨.real,[c.heads*c.T*c.K],"k"⟩,⟨.real,[c.heads*c.T*c.K],"g"⟩]

def consumerInputs (c : Host.Dimensions) : List IOChannel :=
  [⟨.real,[c.heads*c.T*c.K],"q"⟩,⟨.real,[c.heads*c.T*c.K],"g"⟩,
   ⟨.real,[Host.outputSize c],"v"⟩,⟨.fp32,[c.heads*c.NT*c.K*c.V],"h"⟩,
   ⟨.real,[Host.attentionSize c],"attention"⟩]

def producerOutputs (c : Host.Dimensions) : List IOChannel :=
  [⟨.real,[Host.attentionSize c],"attention"⟩,⟨.fp32,[Host.attentionSize c],"attention"⟩]

def consumerOutputs (c : Host.Dimensions) : List IOChannel :=
  [⟨.fp32,[Host.outputSize c],"o"⟩]

def extend {n : Nat} (words : TileIndex [n] → Word) (off : Nat) : Expr :=
  if inside : off<n then .bits (words (⟨off,inside⟩,⟨⟩)) else .bits 0

def producerMemory (c : Host.Dimensions) (values : ChannelValues Word (producerInputs c)) : FPStorage.Memory Expr :=
  fun r off=>if r="q" then extend (values (0 : Fin 3)) off
    else if r="k" then extend (values (1 : Fin 3)) off
    else if r="g" then extend (values (2 : Fin 3)) off
    else .bits 0

def consumerMemory (c : Host.Dimensions) (values : ChannelValues Word (consumerInputs c)) : FPStorage.Memory Expr :=
  fun r off=>if r="q" then extend (values (0 : Fin 5)) off
    else if r="g" then extend (values (1 : Fin 5)) off
    else if r="v" then extend (values (2 : Fin 5)) off
    else if r="h" then extend (values (3 : Fin 5)) off
    else if r="attention" then extend (values (4 : Fin 5)) off
    else .bits 0

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativePorts

/-! Public Real IO for the complete producer and standalone consumer.
All five original source bodies and complete ordinary grids remain explicit.
The producer's rectangular zero initialization and large-K private scratch
are finite fill stages, with no assumed output values or execution callbacks.
-/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def interSource (c : Host.Dimensions) (scale : ℝ) : AlgorithmSource where
  kernel := (SourceBridge.interNew (ProducerGeometry.layout c.producer) ProducerGeometry.regions scale).eraseDType
  projection := by
    simp only [SourceBridge.interNew, TypedInter.original, TypedInter.source, TypedInter.fp32Original, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.surfaceBody,
      Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
    simp only [ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, Except.bind, bind]

def intraSource (c : Host.Dimensions) (scale : ℝ) : AlgorithmSource where
  kernel := (SourceBridge.intraNew (IntraGrid.layout c.producer c.intraBK) IntraGrid.regions scale).eraseDType
  projection := by
    simp only [SourceBridge.intraNew, TypedIntra.original, TypedIntra.source, TypedIntra.fp32Original, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.surfaceBody,
      Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
    simp only [ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, Except.bind, bind]

def splitSource (c : Host.Dimensions) (scale : ℝ) : AlgorithmSource where
  kernel := (SourceBridge.splitNew (SplitGrid.layout c.split) SplitGrid.regions scale).eraseDType
  projection := by
    simp only [SourceBridge.splitNew, TypedSplit.original, TypedSplit.source, TypedSplit.fp32Original, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.surfaceBody,
      Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
    simp only [ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, Except.bind, bind]

def mergeSource (c : Host.Dimensions) : AlgorithmSource where
  kernel := (SourceBridge.mergeNew (MergeGrid.layout c.split) "scratch" "attention").eraseDType
  projection := by
    simp only [SourceBridge.mergeNew, TypedMerge.original, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.surfaceBody,
      Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
    simp only [ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, Except.bind, bind]

def outputSource (c : Host.Dimensions) (scale : ℝ) : AlgorithmSource where
  kernel := (SourceBridge.outputNew (OutputGrid.layout c.consumer) OutputGrid.regions scale).eraseDType
  projection := by
    simp only [SourceBridge.outputNew, TypedOutput.original, TypedOutput.source, TypedOutput.fp32Original, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.surfaceBody,
      Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
    simp only [ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, Except.bind, bind]

def zeroAttention (c : Host.Dimensions) : AlgorithmFill := ⟨"attention",0,Host.attentionSize c,.real 0⟩
def zeroScratch (c : Host.Dimensions) : AlgorithmFill := ⟨"scratch",0,Host.scratchSize c,.real 0⟩

/-- The established Real schedule initializes scratch before inter; the native
entry retains its original later initialization point. Both full schedules
and their respective original execution bridges are proved separately. -/
def producerStages (c : Host.Dimensions) (scale : ℝ) : List AlgorithmFillStage :=
  if c.K≤256 then
    [.fill (zeroAttention c),
      .source (.ordinary (interSource c scale) (ProducerGeometry.grid c.producer)),
      .source (.ordinary (intraSource c scale) (DiagonalGeometry.grid c.producer))]
  else
    [.fill (zeroAttention c),.fill (zeroScratch c),
      .source (.ordinary (interSource c scale) (ProducerGeometry.grid c.producer)),
      .source (.ordinary (splitSource c scale) (SplitGeometry.grid c.split)),
      .source (.ordinary (mergeSource c) (DiagonalGeometry.grid c.split))]

def producer (c : Host.Dimensions) (scale : ℝ) : AlgorithmFillProgramIO where
  nIn := 3
  nOut := 1
  nScr := 1
  bufs := ["q","k","g","attention","scratch"]
  ity := fun _=>.float
  iarity := fun _=>c.heads*c.T*c.K
  ibuf := !["q","k","g"]
  oty := fun _=>.float
  oarity := fun _=>Host.attentionSize c
  obuf := fun _=>"attention"
  obuf_mem := by intro _; simp
  sarity := fun _=>if c.K≤256 then 0 else Host.scratchSize c
  sbuf := fun _=>"scratch"
  iwin := fun _ _ _ _ _ j=>j.val
  imask := fun _ _ _ _ _ _=>True
  owin := fun _ _ _ _ _ j=>j.val
  omask := fun _ _ _ _ _ _=>True
  swin := fun _ _ _ _ _ j=>j.val
  smask := fun _ _ _ _ _ _=>True
  initialUndef := .zero
  stages := fun _=>producerStages c scale

def consumer (c : Host.Dimensions) (scale : ℝ) : AlgorithmFillProgramIO where
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := ["q","g","v","h","attention","o"]
  ity := fun _=>.float
  iarity := ![c.heads*c.T*c.K,c.heads*c.T*c.K,Host.outputSize c,
    c.heads*c.NT*c.K*c.V,Host.attentionSize c]
  ibuf := !["q","g","v","h","attention"]
  oty := fun _=>.float
  oarity := fun _=>Host.outputSize c
  obuf := fun _=>"o"
  obuf_mem := by intro _; simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun _ _ _ _ _ j=>j.val
  imask := fun _ _ _ _ _ _=>True
  owin := fun _ _ _ _ _ j=>j.val
  omask := fun _ _ _ _ _ _=>True
  swin := fun k=>Fin.elim0 k
  smask := fun k=>Fin.elim0 k
  initialUndef := .zero
  stages := fun _=>[.source (.ordinary (outputSource c scale) (OutputGrid.grid c.consumer))]

/-- Zero extension is a mathematical convention outside a finite input
channel. Every operand of an active logical output is proved in range. -/
def extend {n : Nat} (values : Fin n → ℝ) (i : Nat) : ℝ :=
  if h : i<n then values ⟨i,h⟩ else 0

/-- Pure finite data spaces, independent of sources and schedules. -/
abbrev ProducerValues (c : Host.Dimensions) := Fin 3 → Fin (c.heads*c.T*c.K) → ℝ
abbrev ConsumerValues (c : Host.Dimensions) :=
  (channel : Fin 5) → Fin (![c.heads*c.T*c.K,c.heads*c.T*c.K,Host.outputSize c,
    c.heads*c.NT*c.K*c.V,Host.attentionSize c] channel) → ℝ

def producerInputs (c : Host.Dimensions) (_scale : ℝ) (values : ProducerValues c) : Logical.Inputs :=
  ⟨extend (values (0 : Fin 3)),extend (values (1 : Fin 3)),extend (values (2 : Fin 3)),fun _=>0,fun _=>0⟩

def consumerInputs (c : Host.Dimensions) (_scale : ℝ) (values : ConsumerValues c) : RealOutputIO.Inputs :=
  ⟨extend (values (0 : Fin 5)),extend (values (2 : Fin 5)),extend (values (1 : Fin 5)),extend (values (3 : Fin 5)),extend (values (4 : Fin 5))⟩

/-- Causal attention over exactly K logical features. -/
def attention (c : Host.Dimensions) (scale : ℝ) (values : ProducerValues c)
    (_ : Fin 1) (i : Fin (Host.attentionSize c)) : ℝ :=
  Logical.attention c.T c.K c.BT (producerInputs c scale values) scale
    i.divNat.divNat.val i.divNat.modNat.val i.modNat.val

/-- Independent supplied attention/history are retained in the consumer's
logical K history sum plus causal attention/value contraction. -/
def output (c : Host.Dimensions) (scale : ℝ) (values : ConsumerValues c)
    (_ : Fin 1) (i : Fin (Host.outputSize c)) : ℝ :=
  KernelReal.consumerFormula c (consumerInputs c scale values) scale
    i.divNat.divNat i.divNat.modNat i.modNat

#specNonCircular attention avoiding [exec,Producer.direct,Producer.large,producerStages,interSource,intraSource,splitSource,mergeSource]
#specNonCircular output avoiding [exec,OutputGrid.executes,FP.Structural.exec,outputSource]
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealIO

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealIO.LogicalIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
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

def tensor2 (a b : Nat) : (Fin (a*b) → ℝ) ≃ (Fin a → Fin b → ℝ) where
  toFun := fun f i j => f (finProdFinEquiv (i,j))
  invFun := fun f i => f (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
  left_inv := by intro f; funext i; exact congrArg f (finProdFinEquiv.apply_symm_apply i)
  right_inv := by
    intro f
    funext i j
    exact congrArg (fun p => f p.1 p.2) (finProdFinEquiv.symm_apply_apply (i,j))

def coords4 (a b c d : Nat) : (Fin a × Fin b × Fin c × Fin d) ≃ Fin (a*b*c*d) where
  toFun := fun p => finProdFinEquiv (coords3 a b c (p.1,p.2.1,p.2.2.1),p.2.2.2)
  invFun := fun i => let p := finProdFinEquiv.symm i
    let q := (coords3 a b c).symm p.1
    (q.1,q.2.1,q.2.2,p.2)
  left_inv := by
    rintro ⟨h,t,k,v⟩
    simp only [Equiv.symm_apply_apply]
  right_inv := by
    intro i
    change finProdFinEquiv ((coords3 a b c) ((coords3 a b c).symm
      (finProdFinEquiv.symm i).1), (finProdFinEquiv.symm i).2) = i
    rw [Equiv.apply_symm_apply, Equiv.apply_symm_apply]

def tensor4 (a b c d : Nat) : (Fin (a*b*c*d) → ℝ) ≃ (Fin a → Fin b → Fin c → Fin d → ℝ) where
  toFun := fun f h t k v => f (coords4 a b c d (h,t,k,v))
  invFun := fun f i => let p := (coords4 a b c d).symm i
    f p.1 p.2.1 p.2.2.1 p.2.2.2
  left_inv := by intro f; funext i; exact congrArg f ((coords4 a b c d).apply_symm_apply i)
  right_inv := by
    intro f
    funext h t k v
    exact congrArg (fun p => f p.1 p.2.1 p.2.2.1 p.2.2.2)
      ((coords4 a b c d).symm_apply_apply (h,t,k,v))


structure ProducerInputs (c : Host.Dimensions) where
  query : Fin c.heads → Fin c.T → Fin c.K → ℝ
  key : Fin c.heads → Fin c.T → Fin c.K → ℝ
  gate : Fin c.heads → Fin c.T → Fin c.K → ℝ

structure ConsumerInputs (c : Host.Dimensions) where
  query : Fin c.heads → Fin c.T → Fin c.K → ℝ
  gate : Fin c.heads → Fin c.T → Fin c.K → ℝ
  value : Fin c.heads → Fin c.T → Fin c.V → ℝ
  history : Fin c.heads → Fin c.NT → Fin c.K → Fin c.V → ℝ
  attention : Fin c.heads → Fin c.T → Fin c.BT → ℝ

def producerInputs (c : Host.Dimensions) : ProducerValues c ≃ ProducerInputs c where
  toFun := fun values => ⟨tensor3 c.heads c.T c.K (values 0),
    tensor3 c.heads c.T c.K (values 1),tensor3 c.heads c.T c.K (values 2)⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (tensor3 c.heads c.T c.K).symm values.query
    | ⟨1,_⟩ => (tensor3 c.heads c.T c.K).symm values.key
    | ⟨2,_⟩ => (tensor3 c.heads c.T c.K).symm values.gate
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def consumerInputs (c : Host.Dimensions) : ConsumerValues c ≃ ConsumerInputs c where
  toFun := fun values => ⟨tensor3 c.heads c.T c.K (values 0),
    tensor3 c.heads c.T c.K (values 1),tensor3 c.heads c.T c.V (values 2),
    tensor4 c.heads c.NT c.K c.V (values 3),tensor3 c.heads c.T c.BT (values 4)⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (tensor3 c.heads c.T c.K).symm values.query
    | ⟨1,_⟩ => (tensor3 c.heads c.T c.K).symm values.gate
    | ⟨2,_⟩ => (tensor3 c.heads c.T c.V).symm values.value
    | ⟨3,_⟩ => (tensor4 c.heads c.NT c.K c.V).symm values.history
    | ⟨4,_⟩ => (tensor3 c.heads c.T c.BT).symm values.attention
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def outputs3 (a b c : Nat) : (Fin 1 → Fin (a*b*c) → ℝ) ≃ (Fin a → Fin b → Fin c → ℝ) where
  toFun := fun values => tensor3 a b c (values 0)
  invFun := fun values _ => (tensor3 a b c).symm values
  left_inv := by
    intro values
    funext channel j
    have channel_zero : channel = 0 := Fin.eq_zero channel
    subst channel
    exact congrFun ((tensor3 a b c).symm_apply_apply (values 0)) j
  right_inv := fun values => (tensor3 a b c).apply_symm_apply values

/-- These total logical accessors retain exactly the original finite zero
extension. Address conversion belongs to IO, never to the reader formula. -/
def producerArrays (c : Host.Dimensions) (values : ProducerInputs c) :=
  RealIO.producerInputs c 0 ((producerInputs c).symm values)
def consumerArrays (c : Host.Dimensions) (values : ConsumerInputs c) :=
  RealIO.consumerInputs c 0 ((consumerInputs c).symm values)
def producerQuery (c : Host.Dimensions) (values : ProducerInputs c) (h t k : Nat) :=
  (producerArrays c values).query (Logical.featureAddress c.T c.K h t k)
def producerKey (c : Host.Dimensions) (values : ProducerInputs c) (h t k : Nat) :=
  (producerArrays c values).key (Logical.featureAddress c.T c.K h t k)
def producerGate (c : Host.Dimensions) (values : ProducerInputs c) (h t k : Nat) :=
  (producerArrays c values).gate (Logical.featureAddress c.T c.K h t k)
def consumerQuery (c : Host.Dimensions) (values : ConsumerInputs c) (h t k : Nat) :=
  (consumerArrays c values).query (Logical.featureAddress c.T c.K h t k)
def consumerGate (c : Host.Dimensions) (values : ConsumerInputs c) (h t k : Nat) :=
  (consumerArrays c values).gate (Logical.featureAddress c.T c.K h t k)
def consumerHistory (c : Host.Dimensions) (values : ConsumerInputs c) (h chunk k v : Nat) :=
  (consumerArrays c values).history (Geometry.historyAddress c.NT c.K c.V h chunk k v)
def consumerValue (c : Host.Dimensions) (values : ConsumerInputs c) (h t v : Nat) :=
  (consumerArrays c values).value (Geometry.outputAddress c.T c.V h t v)
def consumerAttention (c : Host.Dimensions) (values : ConsumerInputs c) (h t j : Nat) :=
  (consumerArrays c values).attention (h*c.T*c.BT+t*c.BT+j)

def producer (c : Host.Dimensions) (scale : ℝ) :
    LogicalIO (ProducerValues c) (Fin 1 → Fin (Host.attentionSize c) → ℝ)
      (ProducerInputs c) (Fin c.heads → Fin c.T → Fin c.BT → ℝ)
      (RealIO.producer c scale).Implements where
  inputs := producerInputs c
  outputs := outputs3 c.heads c.T c.BT

def consumer (c : Host.Dimensions) (scale : ℝ) :
    LogicalIO (ConsumerValues c) (Fin 1 → Fin (Host.outputSize c) → ℝ)
      (ConsumerInputs c) (Fin c.heads → Fin c.T → Fin c.V → ℝ)
      (RealIO.consumer c scale).Implements where
  inputs := consumerInputs c
  outputs := outputs3 c.heads c.T c.V
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealIO.LogicalIO

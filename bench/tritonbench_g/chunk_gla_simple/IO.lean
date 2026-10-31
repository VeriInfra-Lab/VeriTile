import bench.tritonbench_g.chunk_gla_simple.Proofs.ScalarGraph
import VeriTile.Triton.Memory.RegionalScheduledKernelIO
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Memory.LogicalScheduledKernelIO

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO
open VeriTile Triton FP.Structural Host
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

def inputRegion : Fin 5 → RegionName
  | ⟨0,_⟩ => "q"
  | ⟨1,_⟩ => "k"
  | ⟨2,_⟩ => "v"
  | ⟨3,_⟩ => "h"
  | ⟨4,_⟩ => "g"

def inputSize (c : Shape) : Fin 5 → Nat
  | ⟨0,_⟩ | ⟨1,_⟩ => c.heads*c.T*c.K
  | ⟨2,_⟩ => c.heads*c.T*c.V
  | ⟨3,_⟩ => c.heads*c.NT*c.K*c.V
  | ⟨4,_⟩ => c.heads*c.T

abbrev RealInputs (c : Shape) := (i : Fin 5) → Fin (inputSize c i) → ℝ

def realLaunch (c : Shape) (q : GridIndex (grid c)) : AlgorithmLaunch where
  pids := q.toPids
  numPids := some (grid c).toNumPids

def realSchedule (c : Shape) : List AlgorithmLaunch :=
  (FPGrid.programs c).map (realLaunch c)

/-- Five packed source arrays and the complete contiguous output. The source
uses FP32 storage/computation; dtype erasure selects its mathematical Real
interpretation. There is no private memory scratch or in-place output. -/
def realIO (c : Shape) (scale : ℝ) : UKernelIO where
  kernel := (kernel c scale).eraseDType
  projection := by
    cases hp : (kernel c scale).toAlgorithm? with
    | ok k => simp [ComputeKernel.eraseDType, hp]
    | error e =>
      simp only [kernel,  Bool.false_eq_true, ↓reduceIte,
        ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
        ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Except.bind, bind] at hp
      cases hp
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := ["q","k","v","h","g","o"]
  ity := fun _ => .float
  iarity := inputSize c
  ibuf := inputRegion
  oty := fun _ => .float
  oarity := fun _ => c.heads*c.T*c.V
  obuf := fun _ => "o"
  obuf_mem := by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun t => t.elim0
  smask := fun t => t.elim0

def nativeInputs (c : Shape) : List IOChannel :=
  [⟨.real,[c.heads*c.T*c.K],"q"⟩,⟨.real,[c.heads*c.T*c.K],"k"⟩,
   ⟨.real,[c.heads*c.T*c.V],"v"⟩,⟨.real,[c.heads*c.NT*c.K*c.V],"h"⟩,
   ⟨.real,[c.heads*c.T],"g"⟩]

def nativeLaunch (c : Shape) (q : GridIndex (grid c)) : Invocation NativeGraph.Expr where
  kernel := kernel c ScalarGraph.scaleTag
  backend := FP.ProfiledRewrite.engine ScalarGraph.algebra
  pids := q.toPids
  numPids := some (grid c).toNumPids

/-- Arbitrary symbolic source inputs stay arbitrary. The runtime scale's
existing ABI tag is retained; native scalar lowering is an output observer,
not an input restriction or an alternate kernel. -/
def nativeIO (c : Shape) : ProgramIO NativeGraph.Expr NativeGraph.Expr where
  inputs := nativeInputs c
  outputs := [⟨.fp32,[c.heads,c.T,c.V],"o"⟩]
  scratch := []
  buffers := ["q","k","v","h","g","o"]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => fun j => j.1.val
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val*(c.T*c.V)+j.2.1.val*c.V+j.2.2.1.val)
    (fun o => o.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun o => o.elim0
  scratchMask := fun o => o.elim0
  schedule := fun _ => (FPGrid.programs c).map (nativeLaunch c)
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicMath
open VeriTile Triton Host PublicIO
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math
noncomputable section

def extend {A : Type} {n : Nat} (fallback : A) (xs : Fin n → A) (i : Nat) : A :=
  if hi : i<n then xs ⟨i,hi⟩ else fallback

def arrays (c : Shape) (_scale : ℝ) (xs : RealInputs c) : Inputs :=
  ⟨extend 0 (xs (0 : Fin 5)),extend 0 (xs (1 : Fin 5)),extend 0 (xs (2 : Fin 5)),extend 0 (xs (3 : Fin 5)),extend 0 (xs (4 : Fin 5))⟩

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicMath

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO.Logical
open VeriTile Triton Host
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

structure Inputs (c : Shape) where
  query : Fin c.heads → Fin c.T → Fin c.K → ℝ
  key : Fin c.heads → Fin c.T → Fin c.K → ℝ
  value : Fin c.heads → Fin c.T → Fin c.V → ℝ
  history : Fin c.heads → Fin c.NT → Fin c.K → Fin c.V → ℝ
  gate : Fin c.heads → Fin c.T → ℝ

def inputs (c : Shape) : RealInputs c ≃ Inputs c where
  toFun := fun values => ⟨tensor3 c.heads c.T c.K (values 0),
    tensor3 c.heads c.T c.K (values 1),tensor3 c.heads c.T c.V (values 2),
    tensor4 c.heads c.NT c.K c.V (values 3),tensor2 c.heads c.T (values 4)⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (tensor3 c.heads c.T c.K).symm values.query
    | ⟨1,_⟩ => (tensor3 c.heads c.T c.K).symm values.key
    | ⟨2,_⟩ => (tensor3 c.heads c.T c.V).symm values.value
    | ⟨3,_⟩ => (tensor4 c.heads c.NT c.K c.V).symm values.history
    | ⟨4,_⟩ => (tensor2 c.heads c.T).symm values.gate
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def outputIndices (c : Shape) :
    ((o : Fin 1) × Fin (c.heads*c.T*c.V)) ≃ (Fin c.heads × Fin c.T × Fin c.V) where
  toFun := fun cell => (coords3 c.heads c.T c.V).symm cell.2
  invFun := fun coordinate => ⟨0,coords3 c.heads c.T c.V coordinate⟩
  left_inv := by
    rintro ⟨channel,j⟩
    have channel_zero : channel = 0 := Fin.eq_zero channel
    subst channel
    change (⟨0,coords3 c.heads c.T c.V ((coords3 c.heads c.T c.V).symm j)⟩ :
      ((o : Fin 1) × Fin (c.heads*c.T*c.V)))=⟨0,j⟩
    rw [Equiv.apply_symm_apply]
  right_inv := fun coordinate => (coords3 c.heads c.T c.V).symm_apply_apply coordinate

def arrays (c : Shape) (scale : ℝ) (values : Inputs c) :=
  PublicMath.arrays c scale ((inputs c).symm values)

/-- Tile selection is IO coordinate data; it does not occur in the logical
output coordinates or introduce a new launch/configuration condition. -/
def tileLaunch (c : Shape) (bt : 0<c.BT) (head : Fin c.heads) (time : Fin c.T) (value : Fin c.V) :=
  Host.launchIndex c (Host.owner c bt head time value)

def io (c : Shape) (scale : ℝ) :
    LogicalScheduledKernelIO (Inputs c) (Fin c.heads × Fin c.T × Fin c.V) where
  core := realIO c scale
  schedule := fun _ => realSchedule c
  inputs := inputs c
  indices := outputIndices c
  realOutput := fun _ => rfl

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO.Logical

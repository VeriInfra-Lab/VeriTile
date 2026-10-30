import bench.tritonbench_g.bmm_chunk_fwd.Proofs.KernelInterface
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.HostInputs
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.CleanupReal
import VeriTile.Triton.Memory.AlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
noncomputable section
open Classical
set_option autoImplicit false
set_option linter.unusedSimpArgs false

/-- Finite address envelopes; only the explicitly masked source windows below
are bound. Gaps impose no input-value or allocation requirements. -/
def dataExtent (p : Parameters) : Nat :=
  let v := Host.prepared p.shape (Host.contiguous p.shape)
  p.shape.batch*v.batch+p.shape.seqlen*v.row+p.shape.groups*v.head+p.shape.K*v.column+1

def seqExtent (p : Parameters) : Nat := p.shape.batch*p.seqBatch+p.shape.seqlen*p.seqRow+1

def inputType : Fin 3 → ChanTy
  | ⟨0,_⟩ | ⟨1,_⟩ => .float
  | ⟨2,_⟩ => .int

def inputArity (p : Parameters) : Fin 3 → Nat
  | ⟨0,_⟩ | ⟨1,_⟩ => dataExtent p
  | ⟨2,_⟩ => seqExtent p

def inputBuffer (p : Parameters) : Fin 3 → RegionName
  | ⟨0,_⟩ => p.A
  | ⟨1,_⟩ => p.B
  | ⟨2,_⟩ => Region.cast p.Seq

abbrev RealInputs (p : Parameters) :=
  (i : Fin 3) → Fin (inputArity p i) → (inputType i).carrier

def views (p : Parameters) (c : Host.Config) :=
  RealViews.io Bool.false c.M c.N c.K (parameters p).parameters

/-- Exact union of active input windows over the whole main grid. Sequence
windows are active only in the reached metadata arm; empty output launches
have no input reads. Aliased region names remain permitted. -/
def inputMask (p : Parameters) (c : Host.Config) : Fin 3 → Nat → Prop
  | ⟨0,_⟩,off => p.shape.outputSize≠0 ∧ ∃ pid : Host.PID (parameters p) c,
      ∃ i : TileIndex [c.M,p.shape.K],
      (views p c).aMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i ∧
      (views p c).aWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i=off
  | ⟨1,_⟩,off => p.shape.outputSize≠0 ∧ ∃ pid : Host.PID (parameters p) c,
      ∃ i : TileIndex [p.shape.K,c.N],
      (views p c).bMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i ∧
      (views p c).bWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i=off
  | ⟨2,_⟩,off => p.shape.outputSize≠0 ∧ p.hasSeq=Bool.true ∧
      ((∃ pid : Host.PID (parameters p) c, ∃ i : TileIndex [c.M],
        (views p c).rowIDMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i ∧
        (views p c).rowIDWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i=off) ∨
       (∃ pid : Host.PID (parameters p) c, ∃ i : TileIndex [c.N],
        (views p c).columnIDMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i ∧
        (views p c).columnIDWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i=off))

/-- Mathematical FP32 source projection. The public source identity remains
`Kernels.fp32Source`; no separately maintained Algorithm AST is introduced. -/
def mainSource (p : Parameters) (c : Host.Config) : AlgorithmSource where
  kernel := (Kernels.fp32Source (parameters p) c).eraseDType
  projection := by
    rw [Kernels.fp32Source_eq (parameters p) c rfl rfl]
    have erase : (Kernels.source (parameters p) c).eraseDType =
        RealProgram.kernel Bool.false c.M c.N c.K (parameters p).parameters :=
      RealProgram.storage_erasure .fp32 .fp32 Bool.false .fp32 c.M c.N c.K (parameters p).parameters
    rw [erase]
    exact (views p c).projection

def cleanupSource (p : Parameters) : AlgorithmSource where
  kernel := (Kernels.clearUpper p.Out p.shape.chunk (p.shape.chunks*p.shape.groups)).eraseDType
  projection := by
    simp [Kernels.clearUpper,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?,Except.bind,bind]

def cleanupLaunch (p : Parameters) (pid : Cleanup.PID (parameters p)) : AlgorithmLaunch where
  pids := pid.toPids
  numPids := some (parameters p |>.grid Cleanup.config).toNumPids

def stages (p : Parameters) (c : Host.Config) : List AlgorithmStage :=
  if p.shape.outputSize=0 then [] else
    [.ordinary (mainSource p c) ((parameters p).grid c)] ++
      if p.causal then (Finset.univ.toList : List (Cleanup.PID (parameters p))).map
        (fun pid=>.single (cleanupSource p) (cleanupLaunch p pid)) else []

/-- A/B FP32 mathematical read channels and signed sequence IDs, complete dense
returned output, no private scratch. Ordinary main-grid reads use the same
initial memory; cleanup launches pass the actual predecessor state. -/
def realIO (p : Parameters) (c : Host.Config) : AlgorithmProgramIO where
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := [p.A,p.B,Region.cast p.Seq,p.Out]
  ity := inputType
  iarity := inputArity p
  ibuf := inputBuffer p
  oty := fun _=>.float
  oarity := fun _=>p.shape.outputSize
  obuf := fun _=>p.Out
  obuf_mem := by intro o; simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun _ _ _ _ _ i=>i.val
  imask := fun ch _ _ _ _ i=>inputMask p c ch i.val
  owin := fun _ _ _ _ _ i=>i.val
  omask := fun _ _ _ _ _ _=>True
  swin := fun t=>Fin.elim0 t
  smask := fun t=>Fin.elim0 t
  initialUndef := .preserve
  stages := fun _=>stages p c

def array (p : Parameters) (v : RealInputs p) (which : Fin 2) (off : Nat) : ℝ :=
  if h : off<dataExtent p then
    if which.val=0 then v 0 ⟨off,h⟩ else v 1 ⟨off,h⟩
  else 0

def sequence (p : Parameters) (v : RealInputs p) (off : Nat) : Int :=
  if h : off<seqExtent p then v 2 ⟨off,h⟩ else 0

/-- Source-independent logical arrays for the full dot-product formula. -/
def logicalInputs (p : Parameters) (v : RealInputs p) : Host.Inputs (parameters p) where
  a := fun b r g k=>array p v 0 (Host.inputOffset (Host.prepared p.shape (Host.contiguous p.shape)) b r g k)
  b := fun b r g k=>array p v 1 (Host.inputOffset (Host.prepared p.shape (Host.contiguous p.shape)) b r g k)
  seq := fun b r=>sequence p v (b*p.seqBatch+r*p.seqRow)

def outputColumn (p : Parameters) (off : Nat) := off%p.shape.chunk

def outputRow (p : Parameters) (off : Nat) := off/p.shape.chunk%p.shape.chunk

def outputGroup (p : Parameters) (off : Nat) := off/p.shape.chunk/p.shape.chunk%p.shape.groups

def outputChunk (p : Parameters) (off : Nat) := off/p.shape.chunk/p.shape.chunk/p.shape.groups%p.shape.chunks

def outputBatch (p : Parameters) (off : Nat) := off/p.shape.chunk/p.shape.chunk/p.shape.groups/p.shape.chunks

/-- Independent full mathematical product, signed sequence filtering, padded
zeros and causal upper-triangle zeros. -/
def realFormula (p : Parameters) (v : RealInputs p) (_ : Fin 1) (i : Fin p.shape.outputSize) : ℝ :=
  if p.causal=Bool.true ∧ outputRow p i.val<outputColumn p i.val then 0 else
    Host.expected (parameters p) (logicalInputs p v) (outputBatch p i.val)
      (outputChunk p i.val) (outputGroup p i.val) (outputRow p i.val) (outputColumn p i.val)

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
noncomputable section

/-- Complete storage backing, including every unused address-envelope gap. -/
structure Envelopes (p : Parameters) where
  a : Fin (dataExtent p) → ℝ
  b : Fin (dataExtent p) → ℝ
  sequence : Fin (seqExtent p) → Int

def envelopesEquiv (p : Parameters) : RealInputs p ≃ Envelopes p where
  toFun := fun xs => ⟨xs 0,xs 1,xs 2⟩
  invFun := fun xs => Fin.cases xs.a (Fin.cases xs.b
    (Fin.cases xs.sequence (fun i => i.elim0)))
  left_inv := by intro xs; funext i j; fin_cases i <;> rfl
  right_inv := by intro xs; cases xs; rfl

def Envelopes.tensors {p : Parameters} (backing : Envelopes p) : Host.Inputs (parameters p) :=
  logicalInputs p ((envelopesEquiv p).symm backing)

/-- Mathematical batch/time/group/contraction arrays and signed sequences,
with their complete backing and connection equations. The views are coupled;
arbitrary stride aliases and unused gaps are retained exactly. -/
structure LogicalInputs (p : Parameters) where
  a : Nat → Nat → Nat → Fin p.shape.K → ℝ
  b : Nat → Nat → Nat → Fin p.shape.K → ℝ
  sequence : Nat → Nat → Int
  backing : Envelopes p
  a_connection : a = backing.tensors.a
  b_connection : b = backing.tensors.b
  sequence_connection : sequence = backing.tensors.seq

def graphEquiv (p : Parameters) : Envelopes p ≃ LogicalInputs p where
  toFun := fun backing => ⟨backing.tensors.a,backing.tensors.b,backing.tensors.seq,
    backing,rfl,rfl,rfl⟩
  invFun := LogicalInputs.backing
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨a,b,sequence,backing,ha,hb,hs⟩
    cases ha
    cases hb
    cases hs
    rfl

def logicalInputsEquiv (p : Parameters) : RealInputs p ≃ LogicalInputs p :=
  (envelopesEquiv p).trans (graphEquiv p)

def LogicalInputs.tensors {p : Parameters} (xs : LogicalInputs p) : Host.Inputs (parameters p) where
  a := xs.a
  b := xs.b
  seq := xs.sequence

abbrev Coordinate (p : Parameters) := Fin p.shape.batch × Fin p.shape.chunks ×
  Fin p.shape.groups × Fin p.shape.chunk × Fin p.shape.chunk

def index3 (a b c : Nat) : (Fin a × Fin b × Fin c) ≃ Fin (a*b*c) :=
  (Equiv.prodAssoc _ _ _).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv)
def index4 (a b c d : Nat) : (Fin a × Fin b × Fin c × Fin d) ≃ Fin (a*b*c*d) :=
  (Equiv.prodAssoc _ _ _).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans (index3 (a*b) c d))
def index5 (a b c d e : Nat) : (Fin a × Fin b × Fin c × Fin d × Fin e) ≃ Fin (a*b*c*d*e) :=
  (Equiv.prodAssoc _ _ _).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans (index4 (a*b) c d e))

def coordinateIndex (p : Parameters) : Coordinate p ≃ Fin p.shape.outputSize :=
  (index5 p.shape.batch p.shape.chunks p.shape.groups p.shape.chunk p.shape.chunk).trans
    (finCongr (by simp only [Host.Shape.outputSize,Nat.mul_assoc]))

def logicalOutputsEquiv (p : Parameters) :
    ((o : Fin 1) → Fin p.shape.outputSize → ℝ) ≃ (Coordinate p → ℝ) where
  toFun := fun ys c => ys 0 (coordinateIndex p c)
  invFun := fun f _ i => f ((coordinateIndex p).symm i)
  left_inv := by
    intro ys
    funext o i
    have z : o=0 := Fin.eq_zero o
    subst o
    exact congrArg (ys 0) ((coordinateIndex p).apply_symm_apply i)
  right_inv := by intro f; funext c; exact congrArg f ((coordinateIndex p).symm_apply_apply c)

def logicalIO (p : Parameters) (c : Host.Config) :
    LogicalIO (RealInputs p) ((o : Fin 1) → Fin p.shape.outputSize → ℝ)
      (LogicalInputs p) (Coordinate p → ℝ) (realIO p c).Implements where
  inputs := logicalInputsEquiv p
  outputs := logicalOutputsEquiv p

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO

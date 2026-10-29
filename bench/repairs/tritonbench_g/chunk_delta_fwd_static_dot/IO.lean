import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Kernels
import bench.repairs.tritonbench_g.chunk_delta_fwd.Grid
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.chunk_delta_fwd.RealOutputs
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.NativeReference
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ScalarGraph

/-! Public tensor channels for the complete chunk-delta grid. Weight keeps its
four source strides. Seed and final channels retain their source flags, and
every invocation uses the authoritative quoted FP32 kernel. -/
namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
noncomputable section

def inputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.B*p.H,p.T,p.K],"key"⟩,
   ⟨.real,[p.B*p.H,p.T,p.K],"weight"⟩,
   ⟨.real,[p.B*p.H,p.T,p.V],"value"⟩,
   ⟨.real,[p.B*p.H,p.K,p.V],"initial"⟩]
def outputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.B*p.H,p.NT,p.K,p.V],"history"⟩,
   ⟨.real,[p.B*p.H,p.T,p.V],"corrected"⟩,
   ⟨.real,[p.B*p.H,p.K,p.V],"final"⟩]
abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (inputs p)
abbrev Outputs (α : Type) (p : Parameters) := ChannelValues α (outputs p)

def read (p : Parameters) : (i : Fin (inputs p).length) →
    TileIndex (inputs p)[i].shape → Nat
  | ⟨0,_⟩ => fun j => j.1.val*(p.T*p.K)+j.2.2.1.val+j.2.1.val*p.K
  | ⟨1,_⟩ => fun j => j.1.val/p.H*p.wBatch+j.1.val%p.H*p.wHead+
      j.2.1.val*p.wTime+j.2.2.1.val*p.wDim
  | ⟨2,_⟩ => fun j => j.1.val*(p.T*p.V)+j.2.1.val*p.V+j.2.2.1.val
  | ⟨3,_⟩ => fun j => j.1.val*p.K*p.V+j.2.1.val*p.V+j.2.2.1.val
  | ⟨i+4,h⟩ => False.elim (by simp [inputs] at h)
def inputMask (p : Parameters) : Fin (inputs p).length → Prop
  | ⟨0,_⟩ | ⟨1,_⟩ => 0 < p.V
  | ⟨2,_⟩ => True
  | ⟨3,_⟩ => p.useSeed = Bool.true
  | ⟨i+4,h⟩ => False.elim (by simp [inputs] at h)
def write (p : Parameters) : (o : Fin (outputs p).length) →
    TileIndex (outputs p)[o].shape → Nat
  | ⟨0,_⟩ => fun j => ((j.1.val*p.NT+j.2.1.val)*p.K+j.2.2.1.val)*p.V+j.2.2.2.1.val
  | ⟨1,_⟩ => fun j => (j.1.val*p.T+j.2.1.val)*p.V+j.2.2.1.val
  | ⟨2,_⟩ => fun j => (j.1.val*p.K+j.2.1.val)*p.V+j.2.2.1.val
  | ⟨i+3,h⟩ => False.elim (by simp [outputs] at h)
def outputMask (p : Parameters) : Fin (outputs p).length → Prop
  | ⟨0,_⟩ | ⟨1,_⟩ => True
  | ⟨2,_⟩ => p.storeFinal = Bool.true
  | ⟨i+3,h⟩ => False.elim (by simp [outputs] at h)

def invocation {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (p : Parameters) (pid : PID) : Invocation α where
  kernel := kernel p
  backend := engine (withDot M D)
  clearRegisters := Bool.true
  pids := fun ax => match ax with | 0 => 0 | 1 => pid.1 | _ => pid.2
  numPids := some (fun ax => match ax with | 0 => 1 | 1 => ceilDiv p.V p.BV | _ => p.B*p.H)

/-- Full launch schedule; no one-program substitution for the established grid.
Logical cell dtype is erased Real, while the quoted source fixes FP32 loads,
casts, arithmetic and stores. No private scratch or output aliases are added. -/
def io {α β : Type} (M : Algebra β) (D : DotPrimitive β) (p : Parameters)
    (schedule : List PID) (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs := inputs p
  outputs := outputs p
  buffers := ["key","weight","value","initial","history","corrected","final"]
  scratch := []
  read := fun i _ => read p i
  inputMask := fun i _ _ => inputMask p i
  write := fun o _ => write p o
  outputMask := fun o _ _ => outputMask p o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => schedule.map (invocation M D p)
  encode := encode
  observe := fun _ _ => observe

def key {α : Type} {p : Parameters} (xs : Inputs α p)
    (h : Fin (p.B*p.H)) (t : Fin p.T) (k : Fin p.K) : α :=
  xs ⟨0,by simp [inputs]⟩ (h,t,k,PUnit.unit)
def weight {α : Type} {p : Parameters} (xs : Inputs α p)
    (h : Fin (p.B*p.H)) (t : Fin p.T) (k : Fin p.K) : α :=
  xs ⟨1,by simp [inputs]⟩ (h,t,k,PUnit.unit)
def value {α : Type} {p : Parameters} (xs : Inputs α p)
    (h : Fin (p.B*p.H)) (t : Fin p.T) (v : Fin p.V) : α :=
  xs ⟨2,by simp [inputs]⟩ (h,t,v,PUnit.unit)
def seed {α : Type} {p : Parameters} (xs : Inputs α p)
    (h : Fin (p.B*p.H)) (k : Fin p.K) (v : Fin p.V) : α :=
  xs ⟨3,by simp [inputs]⟩ (h,k,v,PUnit.unit)

def extend {α : Type} [Inhabited α] {n m k : Nat}
    (f : Fin n → Fin m → Fin k → α) (a b c : Nat) : α :=
  if h : a < n ∧ b < m ∧ c < k then f ⟨a,h.1⟩ ⟨b,h.2.1⟩ ⟨c,h.2.2⟩ else default

@[simp] theorem extend_at {α : Type} [Inhabited α] {n m k : Nat}
    (f : Fin n → Fin m → Fin k → α) (a : Fin n) (b : Fin m) (c : Fin k) :
    extend f a.val b.val c.val = f a b c := by simp [extend]

def result {α : Type} {p : Parameters}
    (history : Fin (p.B*p.H) → Fin p.NT → Fin p.K → Fin p.V → α)
    (corrected : Fin (p.B*p.H) → Fin p.T → Fin p.V → α)
    (final : Fin (p.B*p.H) → Fin p.K → Fin p.V → α) : Outputs α p
  | ⟨0,_⟩ => fun j => history j.1 j.2.1 j.2.2.1 j.2.2.2.1
  | ⟨1,_⟩ => fun j => corrected j.1 j.2.1 j.2.2.1
  | ⟨2,_⟩ => fun j => final j.1 j.2.1 j.2.2.1
  | ⟨i+3,h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Interface

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Reference
open VeriTile Triton FP.Structural ChunkDeltaFwd Interface
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
noncomputable section

/-- Logical arrays; extensions outside their stated shapes are never source
input bindings and do not add backing-storage requirements. -/
def realInputs (p : Parameters) (xs : Inputs ℝ p) (head : Nat) : RealMath.Inputs where
  key := extend (key xs) head
  delta := extend (weight xs) head
  value := extend (value xs) head
  seed := extend (seed xs) head

def nativeInputs (p : Parameters) (xs : Inputs (BitVec 32) p) : NativeReference.Inputs where
  key := extend (key xs)
  weight := extend (weight xs)
  value := extend (value xs)
  seed := extend (seed xs)

/-- Existing staged-graph lowering followed by the selected scalar ISA model. -/
def nativeObserve (I : WordPrimitives)
    (value : VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.Expr) : BitVec 32 :=
  (Native.lower value).eval I.isa

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Reference

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.LogicalTensors
open VeriTile Triton FP.Structural ChunkDeltaFwd
noncomputable section

/- Every finite tensor, including inactive optional channels, has a true
coordinate inverse. The original typed memory/schedule/observer is reused. -/
structure Inputs (c : Parameters) where
  key : Fin (c.B*c.H) → Fin (c.T) → Fin (c.K) → ℝ
  weight : Fin (c.B*c.H) → Fin (c.T) → Fin (c.K) → ℝ
  value : Fin (c.B*c.H) → Fin (c.T) → Fin (c.V) → ℝ
  initial : Fin (c.B*c.H) → Fin (c.K) → Fin (c.V) → ℝ

structure Outputs (c : Parameters) where
  history : Fin (c.B*c.H) → Fin (c.NT) → Fin (c.K) → Fin (c.V) → ℝ
  corrected : Fin (c.B*c.H) → Fin (c.T) → Fin (c.V) → ℝ
  final : Fin (c.B*c.H) → Fin (c.K) → Fin (c.V) → ℝ

def inputs (c : Parameters) : Interface.Inputs ℝ c ≃ Inputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K] (values (0 : Fin 4)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K] (values (1 : Fin 4)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V] (values (2 : Fin 4)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V] (values (3 : Fin 4))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K]).symm values.key
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.K]).symm values.weight
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V]).symm values.value
    | ⟨3,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V]).symm values.initial
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def outputs (c : Parameters) : Interface.Outputs ℝ c ≃ Outputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.B*c.H,c.NT,c.K,c.V] (values (0 : Fin 3)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V] (values (1 : Fin 3)),
    Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V] (values (2 : Fin 3))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.NT,c.K,c.V]).symm values.history
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.T,c.V]).symm values.corrected
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.B*c.H,c.K,c.V]).symm values.final
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def realInputs (c : Parameters) (values : Inputs c) (head : Nat) :=
  Reference.realInputs c ((inputs c).symm values) head

def io (c : Parameters) (schedule : List PID) : LogicalProgramIO ℝ (ℝ) (Inputs c) (Outputs c) where
  core := Interface.io Real.algebra Real.dot c schedule id id
  inputs := inputs c
  outputs := outputs c

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.LogicalTensors

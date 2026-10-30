import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelNative
import VeriTile.Triton.Memory.PreparedAlgorithmProgramIO
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Memory.LogicalIO
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Spec

/-! Whole dense logical channels. The flattened lane counts retain the exact
source order: q/k/dq/dk [B*H,T,K], v/do [B*H,T,V], h/dh [B*H,ceil(T/64),K,V],
g [B*H,T], and dg [ceil(K/BK),B*H,T]. No private memory or hidden output exists.
Both launch lists contain the original complete source; only Real explicitly
erases precision. Native keeps the source's FP32 loads/casts and runtime scale.
-/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.IO
open VeriTile Triton FP.Structural Whole KernelNativeCarrier
noncomputable section

def inputRegion : Fin 7 → RegionName := !["q","k","v","h","g","do","dh"]
def outputRegion : Fin 3 → RegionName := !["dq","dk","dg"]
def inputSize (c : Config) (i : Fin 7) : Nat := extent c (inputRegion i)
def outputSize (c : Config) (o : Fin 3) : Nat := extent c (outputRegion o)
abbrev Inputs (α : Type) (c : Config) := (i : Fin 7) → Fin (inputSize c i) → α
abbrev Outputs (α : Type) (c : Config) := (o : Fin 3) → Fin (outputSize c o) → α

def realJob (c : Config) (a : CTA c) : PreparedAlgorithmInvocation where
  source := ⟨(kernel c).eraseDType,by
    have hp : (kernel c).toAlgorithm? = .ok (kernel c).toAlgKernel :=
      RealSource.projects .fp32 .fp32 .fp32 .fp32 Bool.false
        "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
        c.layout.s_k_h c.layout.s_k_t c.layout.s_v_h c.layout.s_v_t
        c.layout.s_h_h c.layout.s_h_t c.scale c.T c.K c.V 64 c.BK c.BV c.NT
    simp only [ComputeKernel.eraseDType,hp,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]⟩
  launch := {
    pids:=fun i=>if i=0 then a.key.val else if i=1 then a.time.val else if i=2 then a.head.val else 0
    numPids:=some (fun i=>if i=0 then c.NK else if i=1 then c.NT else if i=2 then c.heads else 1)}
  clearRegisters := Bool.true
  zeroUndef := Bool.true

def realIO (c : Config) (order : List (CTA c)) : PreparedAlgorithmProgramIO where
  nIn:=7
  nOut:=3
  nScr:=0
  bufs:=regions
  ity:=fun _=>.float
  iarity:=inputSize c
  ibuf:=inputRegion
  oty:=fun _=>.float
  oarity:=outputSize c
  obuf:=outputRegion
  obuf_mem:=by intro o; fin_cases o <;> simp [outputRegion,regions]
  sarity:=fun i=>Fin.elim0 i
  sbuf:=fun i=>Fin.elim0 i
  iwin:=fun _ _ _ _ _ j=>j.val
  imask:=fun _ _ _ _ _ _=>True
  owin:=fun _ _ _ _ _ j=>j.val
  omask:=fun _ _ _ _ _ _=>True
  swin:=fun i=>Fin.elim0 i
  smask:=fun i=>Fin.elim0 i
  schedule:=fun _=>order.map (realJob c)

def inputDType : Fin 7 → TileDType := ![.fp32,.fp32,.fp32,.real,.real,.fp32,.real]
def outputDType : Fin 3 → TileDType := ![.fp32,.fp32,.real]
def nativeInputs (c : Config) : List IOChannel :=
  [⟨.fp32,[inputSize c 0],"q"⟩,⟨.fp32,[inputSize c 1],"k"⟩,
   ⟨.fp32,[inputSize c 2],"v"⟩,⟨.real,[inputSize c 3],"h"⟩,
   ⟨.real,[inputSize c 4],"g"⟩,⟨.fp32,[inputSize c 5],"do"⟩,
   ⟨.real,[inputSize c 6],"dh"⟩]
def nativeOutputs (c : Config) : List IOChannel :=
  [⟨.fp32,[outputSize c 0],"dq"⟩,⟨.fp32,[outputSize c 1],"dk"⟩,
   ⟨.real,[outputSize c 2],"dg"⟩]

def nativeJob (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (a : CTA c) : Invocation Node where
  kernel:=Whole.kernel (nativeConfig c)
  backend:=FP.ProfiledRewrite.engine (Typed.withDot (algebra isa route scale) dotModel)
  clearRegisters:=Bool.true
  pids:=fun i=>if i=0 then a.key.val else if i=1 then a.time.val else if i=2 then a.head.val else 0
  numPids:=some (fun i=>if i=0 then c.NK else if i=1 then c.NT else if i=2 then c.heads else 1)

/-- Symbolic inputs remain arbitrary native terms, as in the retained raw
theorem. Output evaluation is a separate shared Observes judgment, so no leaf
restriction or loss of raw word cases is hidden in input encoding. -/
def nativeIO (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) : ProgramIO Node Node where
  scratch:=[]
  inputs:=nativeInputs c
  outputs:=nativeOutputs c
  buffers:=regions
  read:=by
    intro i vals j
    change Fin 7 at i
    by_cases h0 : i=0
    · subst i; exact j.1.val
    by_cases h1 : i=1
    · subst i; exact j.1.val
    by_cases h2 : i=2
    · subst i; exact j.1.val
    by_cases h3 : i=3
    · subst i; exact j.1.val
    by_cases h4 : i=4
    · subst i; exact j.1.val
    by_cases h5 : i=5
    · subst i; exact j.1.val
    have hlast : i=6 := by omega
    subst i; exact j.1.val
  inputMask:=fun _ _ _=>True
  write:=by
    intro o vals j
    change Fin 3 at o
    by_cases h0 : o=0
    · subst o; exact j.1.val
    by_cases h1 : o=1
    · subst o; exact j.1.val
    have hlast : o=2 := by omega
    subst o; exact j.1.val
  outputMask:=fun _ _ _=>True
  scratchWrite:=fun k=>Fin.elim0 k
  scratchMask:=fun k=>Fin.elim0 k
  schedule:=fun _=>order.map (nativeJob isa route scale c)
  encode:=id
  observe:=fun _ _=>id

def nativeValues (c : Config) (vals : ChannelValues Node (nativeInputs c)) : Inputs Node c := by
  intro i j
  by_cases h0 : i=0
  · subst i; exact vals (0 : Fin 7) (j,PUnit.unit)
  by_cases h1 : i=1
  · subst i; exact vals (1 : Fin 7) (j,PUnit.unit)
  by_cases h2 : i=2
  · subst i; exact vals (2 : Fin 7) (j,PUnit.unit)
  by_cases h3 : i=3
  · subst i; exact vals (3 : Fin 7) (j,PUnit.unit)
  by_cases h4 : i=4
  · subst i; exact vals (4 : Fin 7) (j,PUnit.unit)
  by_cases h5 : i=5
  · subst i; exact vals (5 : Fin 7) (j,PUnit.unit)
  have hlast : i=6 := by omega
  subst i; exact vals (6 : Fin 7) (j,PUnit.unit)
def nativeResult (c : Config) (out : Outputs Node c) : ChannelValues Node (nativeOutputs c) := by
  intro o j
  change Fin 3 at o
  by_cases h0 : o=0
  · subst o; exact out 0 j.1
  by_cases h1 : o=1
  · subst o; exact out 1 j.1
  have hlast : o=2 := by omega
  subst o; exact out 2 j.1

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.IO

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.PairConnection
open VeriTile Triton FP.Structural Whole
noncomputable section

def kernel (c : Config) : ComputeKernel :=
  TypedSource.tailBoundCSE
    "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
    c.layout.s_k_h c.layout.s_k_t c.layout.s_v_h c.layout.s_v_t
    c.layout.s_h_h c.layout.s_h_t c.scale c.T c.K c.V 64 c.BK c.BV c.NT
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.PairConnection

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.PairIO
open VeriTile Triton FP.Structural Whole KernelNativeCarrier
noncomputable section

def invocation (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (pid : CTA c) : Invocation Node :=
  { IO.nativeJob isa route scale c pid with kernel := PairConnection.kernel (nativeConfig c) }

def candidate (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) :
    ProgramIO.Variant (IO.nativeIO isa route scale c order) where
  schedule := fun _ => order.map (invocation isa route scale c)
  observe := (IO.nativeIO isa route scale c order).observe
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.PairIO

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Target
open VeriTile Triton FP.Structural Whole KernelNativeCarrier
noncomputable section

/-- Pure array extension outside the logical input domains. Source masks prove
that these default values cannot influence an active result. -/
def get {α : Type} [Inhabited α] (c : Config) (vals : IO.Inputs α c) (i : Fin 7) (n : Nat) : α :=
  if h : n<IO.inputSize c i then vals i ⟨n,h⟩ else default
def inputs {α : Type} [Inhabited α] (c : Config) (vals : IO.Inputs α c) (r : RegionName) : Nat → α :=
  if r="q" then get c vals 0 else if r="k" then get c vals 1 else
  if r="v" then get c vals 2 else if r="h" then get c vals 3 else
  if r="g" then get c vals 4 else if r="do" then get c vals 5 else
  if r="dh" then get c vals 6 else fun _=>default

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Target

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Logical
open VeriTile Triton Whole
noncomputable section

def index3 (a b c : Nat) : (Fin a × Fin b × Fin c) ≃ Fin (a*b*c) :=
  (Equiv.prodAssoc _ _ _).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv)
def index4 (a b c d : Nat) : (Fin a × Fin b × Fin c × Fin d) ≃ Fin (a*b*c*d) :=
  (Equiv.prodAssoc _ _ _).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans (index3 (a*b) c d))

structure Inputs (c : Config) where
  query : (Fin c.heads × Fin c.T × Fin c.K) → ℝ
  key : (Fin c.heads × Fin c.T × Fin c.K) → ℝ
  value : (Fin c.heads × Fin c.T × Fin c.V) → ℝ
  history : (Fin c.heads × Fin c.NT × Fin c.K × Fin c.V) → ℝ
  gate : (Fin c.heads × Fin c.T) → ℝ
  outputGrad : (Fin c.heads × Fin c.T × Fin c.V) → ℝ
  historyGrad : (Fin c.heads × Fin c.NT × Fin c.K × Fin c.V) → ℝ

def inputsEquiv (c : Config) : IO.Inputs ℝ c ≃ Inputs c where
  toFun := fun vals => ⟨fun p => vals (0 : Fin 7) (index3 c.heads c.T c.K p),
    fun p => vals (1 : Fin 7) (index3 c.heads c.T c.K p),
    fun p => vals (2 : Fin 7) (index3 c.heads c.T c.V p),
    fun p => vals (3 : Fin 7) (index4 c.heads c.NT c.K c.V p),
    fun p => vals (4 : Fin 7) (finProdFinEquiv p),
    fun p => vals (5 : Fin 7) (index3 c.heads c.T c.V p),
    fun p => vals (6 : Fin 7) (index4 c.heads c.NT c.K c.V p)⟩
  invFun := fun xs => (Fin.cases (fun j => xs.query ((index3 c.heads c.T c.K).symm j)) (Fin.cases (fun j => xs.key ((index3 c.heads c.T c.K).symm j)) (Fin.cases (fun j => xs.value ((index3 c.heads c.T c.V).symm j)) (Fin.cases (fun j => xs.history ((index4 c.heads c.NT c.K c.V).symm j)) (Fin.cases (fun j => xs.gate ((finProdFinEquiv).symm j)) (Fin.cases (fun j => xs.outputGrad ((index3 c.heads c.T c.V).symm j)) (Fin.cases (fun j => xs.historyGrad ((index4 c.heads c.NT c.K c.V).symm j)) (fun i => i.elim0))))))))
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    · exact congrArg (vals (0 : Fin 7)) ((index3 c.heads c.T c.K).apply_symm_apply j)
    · exact congrArg (vals (1 : Fin 7)) ((index3 c.heads c.T c.K).apply_symm_apply j)
    · exact congrArg (vals (2 : Fin 7)) ((index3 c.heads c.T c.V).apply_symm_apply j)
    · exact congrArg (vals (3 : Fin 7)) ((index4 c.heads c.NT c.K c.V).apply_symm_apply j)
    · exact congrArg (vals (4 : Fin 7)) ((finProdFinEquiv).apply_symm_apply j)
    · exact congrArg (vals (5 : Fin 7)) ((index3 c.heads c.T c.V).apply_symm_apply j)
    · exact congrArg (vals (6 : Fin 7)) ((index4 c.heads c.NT c.K c.V).apply_symm_apply j)
  right_inv := by
    intro xs
    change Inputs.mk (fun p => xs.query ((index3 c.heads c.T c.K).symm (index3 c.heads c.T c.K p)))
      (fun p => xs.key ((index3 c.heads c.T c.K).symm (index3 c.heads c.T c.K p)))
      (fun p => xs.value ((index3 c.heads c.T c.V).symm (index3 c.heads c.T c.V p)))
      (fun p => xs.history ((index4 c.heads c.NT c.K c.V).symm (index4 c.heads c.NT c.K c.V p)))
      (fun p => xs.gate ((finProdFinEquiv).symm (finProdFinEquiv p)))
      (fun p => xs.outputGrad ((index3 c.heads c.T c.V).symm (index3 c.heads c.T c.V p)))
      (fun p => xs.historyGrad ((index4 c.heads c.NT c.K c.V).symm (index4 c.heads c.NT c.K c.V p))) = xs
    simp only [Equiv.symm_apply_apply]

structure Outputs (c : Config) where
  dq : (Fin c.heads × Fin c.T × Fin c.K) → ℝ
  dk : (Fin c.heads × Fin c.T × Fin c.K) → ℝ
  dg : (Fin c.NK × Fin c.heads × Fin c.T) → ℝ

def outputsEquiv (c : Config) : IO.Outputs ℝ c ≃ Outputs c where
  toFun := fun vals => ⟨fun p => vals (0 : Fin 3) (index3 c.heads c.T c.K p), fun p => vals (1 : Fin 3) (index3 c.heads c.T c.K p), fun p => vals (2 : Fin 3) (index3 c.NK c.heads c.T p)⟩
  invFun := fun xs => (Fin.cases (fun j => xs.dq ((index3 c.heads c.T c.K).symm j)) (Fin.cases (fun j => xs.dk ((index3 c.heads c.T c.K).symm j)) (Fin.cases (fun j => xs.dg ((index3 c.NK c.heads c.T).symm j)) (fun i => i.elim0))))
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    · exact congrArg (vals (0 : Fin 3)) ((index3 c.heads c.T c.K).apply_symm_apply j)
    · exact congrArg (vals (1 : Fin 3)) ((index3 c.heads c.T c.K).apply_symm_apply j)
    · exact congrArg (vals (2 : Fin 3)) ((index3 c.NK c.heads c.T).apply_symm_apply j)
  right_inv := by
    intro xs
    change Outputs.mk (fun p => xs.dq ((index3 c.heads c.T c.K).symm (index3 c.heads c.T c.K p)))
      (fun p => xs.dk ((index3 c.heads c.T c.K).symm (index3 c.heads c.T c.K p)))
      (fun p => xs.dg ((index3 c.NK c.heads c.T).symm (index3 c.NK c.heads c.T p))) = xs
    simp only [Equiv.symm_apply_apply]

def Inputs.tensors {c : Config} (xs : Inputs c) : Spec.Tensors :=
  Spec.views c (Target.inputs c ((inputsEquiv c).symm xs))

def io (c : Config) (order : List (CTA c)) :
    LogicalIO (IO.Inputs ℝ c) (IO.Outputs ℝ c) (Inputs c) (Outputs c)
      (IO.realIO c order).Implements where
  inputs := inputsEquiv c
  outputs := outputsEquiv c

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Logical

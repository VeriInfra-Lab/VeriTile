import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeSerial
import VeriTile.Triton.Float.ObservedProgramIO

/-! Public native IO. Finite windows enumerate the actual input and output
addresses of every original grid lane, including the guarded dummy programs.
No positivity, dtype-tag or numerical guard is added to the original configs.
Repeated read addresses are bound simultaneously to the same decoded value. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
open scoped Classical

def innerInputs (c : HostFP.Inner) : List IOChannel :=
  [⟨.real,[(collect (keyReads c)).length],"k"⟩,
   ⟨.real,[(collect (valueReads c)).length],"v"⟩,
   ⟨.real,[(collect (gateReads c)).length],"g"⟩,
   ⟨.fp32,[(collect (seedReads c)).length],"h0"⟩]
def innerOutputs (c : HostFP.Inner) : List IOChannel :=
  [⟨.real,[(collect (historyWrites c)).length],"h"⟩,
   ⟨.fp32,[(collect (finalWrites c)).length],"ht"⟩]
def prefixInputs (c : HostFP.Cumulative) : List IOChannel :=
  [⟨.fp32,[(collect (prefixCells c)).length],"s"⟩]
def prefixOutputs (c : HostFP.Cumulative) : List IOChannel :=
  [⟨.fp32,[(collect (prefixCells c)).length],"o"⟩]

/-- Source precision and literal region roles are fixed by innerSource.
All input channels are read-only; no private scratch or output alias is hidden.
History is the actual erased Real channel with FP32-profile cast; final is FP32.
The following proofs connect this full schedule to the original merged grids. -/
def innerIO (c : HostFP.Inner) : ProgramIO NativeGraph.Expr NativeGraph.Expr where
  inputs := innerInputs c
  outputs := innerOutputs c
  scratch := []
  buffers := ["k","v","g","h0","h","ht"]
  read := fun i _ => match i with
    | ⟨0,_⟩ => fun j => (collect (keyReads c))[j.1]
    | ⟨1,_⟩ => fun j => (collect (valueReads c))[j.1]
    | ⟨2,_⟩ => fun j => (collect (gateReads c))[j.1]
    | ⟨3,_⟩ => fun j => (collect (seedReads c))[j.1]
    | ⟨n+4,h⟩ => False.elim (by change n+4<4 at h; omega)
  inputMask := fun _ _ _ => True
  write := fun o _ => match o with
    | ⟨0,_⟩ => fun j => (collect (historyWrites c))[j.1]
    | ⟨1,_⟩ => fun j => (collect (finalWrites c))[j.1]
    | ⟨n+2,h⟩ => False.elim (by change n+2<2 at h; omega)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => (allJobs c.plans).map (invocation (NativeGrid.innerSource c))
  encode := id
  observe := fun _ _ => id

/-- The prefix retains every original autotune plan and all guarded dummies;
BS is not restricted to the checked Tensor adapter's usual choices. -/
def prefixIO (c : HostFP.Cumulative) : ProgramIO NativeGraph.Expr NativeGraph.Expr where
  inputs := prefixInputs c
  outputs := prefixOutputs c
  scratch := []
  buffers := ["s","o"]
  read := fun i _ => match i with
    | ⟨0,_⟩ => fun j => (collect (prefixCells c))[j.1]
    | ⟨n+1,h⟩ => False.elim (by change n+1<1 at h; omega)
  inputMask := fun _ _ _ => True
  write := fun o _ => match o with
    | ⟨0,_⟩ => fun j => (collect (prefixCells c))[j.1]
    | ⟨n+1,h⟩ => False.elim (by change n+1<1 at h; omega)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => (allJobs c.plans).map (invocation (NativeGrid.prefixSource c))
  encode := id
  observe := fun _ _ => id

/-- Finite logical input lookup. A repeated address is pinned consistently by
the shared InputBinding; no infinite raw-memory snapshot is an input premise. -/
def lookup (addresses : List Nat) (values : Fin addresses.length → NativeGraph.Expr)
    (offset : Nat) : NativeGraph.Expr :=
  if found : ∃ i : Fin addresses.length,addresses[i]=offset then values (Classical.choose found)
  else default

def innerInput (c : HostFP.Inner) (x : (innerIO c).Inputs) : State NativeGraph.Expr :=
  { emptyState with mem := fun r off =>
    if r="k" then .mk .real (lookup (collect (keyReads c)) (fun i=>x (0 : Fin 4) (i,PUnit.unit)) off)
    else if r="v" then .mk .real (lookup (collect (valueReads c)) (fun i=>x (1 : Fin 4) (i,PUnit.unit)) off)
    else if r="g" then .mk .real (lookup (collect (gateReads c)) (fun i=>x (2 : Fin 4) (i,PUnit.unit)) off)
    else if r="h0" then .mk .fp32 (lookup (collect (seedReads c)) (fun i=>x (3 : Fin 4) (i,PUnit.unit)) off)
    else emptyState.mem r off }

def prefixInput (c : HostFP.Cumulative) (x : (prefixIO c).Inputs) : State NativeGraph.Expr :=
  { emptyState with mem := fun r off =>
    if r="s" then .mk .fp32 (lookup (collect (prefixCells c)) (fun i=>x (0 : Fin 1) (i,PUnit.unit)) off)
    else emptyState.mem r off }

def innerGraphStores (c : HostFP.Inner) (x : (innerIO c).Inputs) :=
  (allJobs c.plans).flatMap fun j => innerStores c j (innerInput c x)
def prefixGraphStores (c : HostFP.Cumulative) (x : (prefixIO c).Inputs) :=
  (allJobs c.plans).flatMap fun j => prefixStores c j (prefixInput c x)

/-- Independent native arithmetic recurrence: ordered dot accumulator, gates,
seed addition and explicit casts. This target never executes a source kernel. -/
def innerGraph (c : HostFP.Inner) (x : (innerIO c).Inputs) : (innerIO c).Outputs :=
  fun o => match o with
  | ⟨0,_⟩ => fun j => (NativeStores.apply (innerGraphStores c x) emptyState.mem "h"
      ((collect (historyWrites c))[j.1])).read .real
  | ⟨1,_⟩ => fun j => (NativeStores.apply (innerGraphStores c x) emptyState.mem "ht"
      ((collect (finalWrites c))[j.1])).read .fp32
  | ⟨n+2,h⟩ => False.elim (by change n+2<2 at h; omega)

/-- The independent triangular-dot prefix recurrence, with all source casts. -/
def prefixGraph (c : HostFP.Cumulative) (x : (prefixIO c).Inputs) : (prefixIO c).Outputs :=
  fun o => match o with
  | ⟨0,_⟩ => fun j => (NativeStores.apply (prefixGraphStores c x) emptyState.mem "o"
    ((collect (prefixCells c))[j.1])).read .fp32
  | ⟨n+1,h⟩ => False.elim (by change n+1<1 at h; omega)

#specNonCircular innerGraph avoiding [FP.Structural.exec,FP.Structural.run,runSchedule,HostFP.batches]
#specNonCircular prefixGraph avoiding [FP.Structural.exec,FP.Structural.run,runSchedule,HostFP.batches]
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface

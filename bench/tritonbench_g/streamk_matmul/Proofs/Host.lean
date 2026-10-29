import bench.tritonbench_g.streamk_matmul.HostDefinitions
import bench.tritonbench_g.streamk_matmul.Proofs.GeneratedSource
import bench.tritonbench_g.streamk_matmul.Proofs.Reference
namespace VeriTile.Bench.TritonBenchG.StreamK

/-- Exact contiguous host strides and GROUP_M=8. This profile explicitly picks
signed i32 scalar specialization; the source theorem also handles arbitrary
Config strides/GROUP_M. Native specialization/launch legality is separate. -/
def hostConfig (h : HostInput) (bases : Region→Address) : Config where
 bm:=h.bm
 bn:=h.bn
 bk:=h.bk
 base:=bases
 firstGrid:=h.programs
 fullGrid:=(hostPlan h).blockingTiles
 param:=fun p=>BitVec.ofNat 32 (match p with
  | .m=>h.m | .n=>h.n | .k=>h.k | .bm=>h.bm | .bn=>h.bn | .bk=>h.bk
  | .groupM=>8 | .strideAM=>h.k | .strideAK=>1 | .strideBK=>h.n | .strideBN=>1
  | .strideCM=>h.n | .strideCN=>1
  | .fullIters=>(hostPlan h).fullIterations | .partialIters=>(hostPlan h).partialIterations
  | .itersPerTile=>(hostPlan h).iterations | .streamTiles=>(hostPlan h).streamTiles)

/-- torch.zeros initializes exactly the allocated lock cells; cells outside
that allocation retain their caller-supplied interpretation. torch.empty C
adds no zero-value premise. Logical regions need not be assumed disjoint by
the value theorem (native typed-byte/alias refinement is separate). -/
def allocateLocks (h : HostInput) (bases : Region→Address) (mem : Memory V) : Memory V :=
 {mem with locks:=fun a=>if a.1=(bases .locks).1 ∧
    0≤a.2-(bases .locks).2 ∧
    (a.2-(bases .locks).2).toNat<(hostPlan h).streamTiles then 0 else mem.locks a}

theorem allocated_lock_zero (h : HostInput) (bases : Region→Address) (mem : Memory V)
 (i : Nat) (hi:i<(hostPlan h).streamTiles) :
 (allocateLocks h bases mem).locks ((bases .locks).1,(bases .locks).2+Int.ofNat i)=0 := by
 have hc : ((bases .locks).1=(bases .locks).1 ∧
    0≤((bases .locks).2+Int.ofNat i)-(bases .locks).2 ∧
    (((bases .locks).2+Int.ofNat i)-(bases .locks).2).toNat<(hostPlan h).streamTiles) := by
  constructor
  · rfl
  constructor
  · have hn : 0≤Int.ofNat i := Int.natCast_nonneg i
    omega
  · have hs : ((bases .locks).2+Int.ofNat i)-(bases .locks).2=Int.ofNat i := by omega
    rw [hs]
    exact hi
 change (if _ then _ else _)=_
 exact if_pos hc

def allDone (grid : Nat) (control : Nat→Control) : Bool :=
 (List.range grid).all fun p=>(control p).todo.isEmpty && !(control p).fault

def restart (source : List Command) (b : Backend V) (s : State V) : State V :=
 {s with control:=fun _=>initialControl source,regs:=fun _ _ _=>b.zero}
def referenceRestart (r : Reference) : Reference := {r with regs:=fun _ _ _=>.zero}

def firstRun (c : Config) (b : Backend V) (schedule : List Nat) (mem : Memory V) :=
 run c b c.firstGrid schedule (initial originalSource.first b mem)
def secondRun (c : Config) (b : Backend V) (schedule : List Nat) (s : State V) :=
 run c b c.fullGrid schedule (restart originalSource.fullWave b s)

/-- Stream ordering is explicit. A blocked/faulted finite first-wave schedule
returns none instead of pretending the second launch ran or the first ended. -/
def hostRun (c : Config) (b : Backend V) (first second : List Nat) (mem : Memory V) : Option (State V) :=
 let s:=(firstRun c b first mem).2
 if allDone c.firstGrid s.control then
   let final := (secondRun c b second s).2
   if allDone c.fullGrid final.control then some final else none
 else none

def firstControl (c : Config) (schedule : List Nat) (locks : Address→I32) :=
 controlRun c c.firstGrid schedule (controlInitial originalSource.first locks)
def secondControl (c : Config) (schedule : List Nat) (locks : Address→I32) :=
 controlRun c c.fullGrid schedule (controlInitial originalSource.fullWave locks)

def firstReference (c : Config) (schedule : List Nat) (locks : Address→I32) : Reference :=
 replay c (firstControl c schedule locks).1 (referenceInitial locks)
def hostReference (c : Config) (first second : List Nat) (locks : Address→I32) : Option Reference :=
 let ctl:=firstControl c first locks
 let r:=firstReference c first locks
 if allDone c.firstGrid ctl.2.programs then
   let next := secondControl c second ctl.2.locks
   if allDone c.fullGrid next.2.programs then
     some (replay c next.1 (referenceRestart r)) else none
 else none

theorem first_related (c : Config) (b : Backend V) (schedule : List Nat) (mem : Memory V) :
 Related b mem.data (firstRun c b schedule mem).2 (firstReference c schedule mem.locks) := by
 have hr:=run_related c b mem.data c.firstGrid schedule _ _ (related_initial originalSource.first b mem)
 have ht:=trace_independent originalSource.first c b c.firstGrid schedule mem
 simpa only [firstRun,firstReference,firstControl,controlTrace,ht] using hr

theorem first_control_projection (c : Config) (b : Backend V) (schedule : List Nat) (mem : Memory V) :
 (firstControl c schedule mem.locks).2=projectControl (firstRun c b schedule mem).2 :=
 congrArg Prod.snd (run_control c b c.firstGrid schedule (initial originalSource.first b mem))

theorem restart_related (source : List Command) (b : Backend V) (mem : Address→V)
 (s : State V) (r : Reference) (h:Related b mem s r) :
 Related b mem (restart source b s) (referenceRestart r) :=
 ⟨h.data,h.locks,fun _ _ _=>rfl⟩

theorem second_related (c : Config) (b : Backend V) (schedule : List Nat) (mem : Address→V)
 (s : State V) (r : Reference) (h:Related b mem s r) :
 Related b mem (secondRun c b schedule s).2
   (replay c (secondControl c schedule s.mem.locks).1 (referenceRestart r)) := by
 have hr:=run_related c b mem c.fullGrid schedule _ _ (restart_related originalSource.fullWave b mem s r h)
 have ht:=trace_independent originalSource.fullWave c b c.fullGrid schedule s.mem
 have ht' : (run c b c.fullGrid schedule (restart originalSource.fullWave b s)).1 =
     (secondControl c schedule s.mem.locks).1 := ht
 rw [ht'] at hr
 exact hr

theorem second_control_projection (c : Config) (b : Backend V) (schedule : List Nat) (s : State V) :
 (secondControl c schedule s.mem.locks).2=projectControl (secondRun c b schedule s).2 :=
 congrArg Prod.snd (run_control c b c.fullGrid schedule (restart originalSource.fullWave b s))

/-- Pure symbolic expected state from initial memory and integer control only;
no execution outcome or accumulator is a premise. -/
theorem host_value_contract (c : Config) (b : Backend V) (first second : List Nat) (mem : Memory V) :
 match hostRun c b first second mem,hostReference c first second mem.locks with
 | none,none => True
 | some s,some r => Related b mem.data s r
 | _,_ => False := by
 have hc:=first_control_projection c b first mem
 have hr:=first_related c b first mem
 unfold hostRun hostReference
 dsimp only
 rw [hc]
 simp only [projectControl]
 by_cases done : allDone c.firstGrid (firstRun c b first mem).2.control = true
 · simp only [done,ite_true]
   have hs:=second_control_projection c b second (firstRun c b first mem).2
   rw [hs]
   simp only [projectControl]
   by_cases done2 : allDone c.fullGrid (secondRun c b second (firstRun c b first mem).2).2.control=true
   · simp only [done2,ite_true]
     exact second_related c b second mem.data (firstRun c b first mem).2
       (firstReference c first mem.locks) hr
   · simp only [done2,ite_false]
     trivial
 · simp only [done,ite_false]
   trivial

#axiomsClean allocated_lock_zero
#axiomsClean first_related
#axiomsClean second_related
#axiomsClean host_value_contract
#specNonCircular hostReference avoiding [run,step,execute,firstRun,secondRun,hostRun]
end VeriTile.Bench.TritonBenchG.StreamK

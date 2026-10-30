import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostNative
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Packing

namespace VeriTile.Bench.Repairs.ParallelAttention.Physical
noncomputable section
set_option maxHeartbeats 500000
open HostAPI

abbrev Heap := Nat → BitVec 8

def observe (h : Heap) : Memory FP32Word := fun a=>Packing.value (fun k=>h (4*a+k.val))

/-- A real four-byte FP32 store at an aligned physical word address. -/
def store (h : Heap) (a : Nat) (x : FP32Word) : Heap := fun q=>
  if q/4=a then Packing.word x ⟨q%4,Nat.mod_lt _ (by decide)⟩ else h q

theorem byte_observe (h : Heap) (q : Nat) :
    Packing.word (observe h (q/4)) ⟨q%4,Nat.mod_lt _ (by decide)⟩=h q := by
  rw [observe,Packing.word_value]
  change h (4*(q/4)+q%4)=h q
  congr 1
  omega

theorem observe_injective : Function.Injective observe := by
  intro h g eq
  funext q
  rw [←byte_observe h q,←byte_observe g q,eq]

theorem observe_store (h : Heap) (a : Nat) (x : FP32Word) :
    observe (store h a x)=Function.update (observe h) a x := by
  funext i
  by_cases hit:i=a
  · subst i
    simp only [Function.update_self]
    have words : (fun k : Fin 4=>store h a x (4*a+k.val))=Packing.word x := by
      funext k
      have hk:=k.isLt
      simp [store,show (4*a+k.val)/4=a by omega,show (4*a+k.val)%4=k.val by omega]
    change Packing.value _=x
    rw [words,Packing.value_word]
  · rw [Function.update_of_ne hit]
    unfold observe
    congr 1
    funext k
    have hk:=k.isLt
    simp [store,show (4*i+k.val)/4=i by omega,hit]

/-- Each event reads the evolving byte heap before performing its own store.
No completed tensor or initial-memory numerical snapshot is an argument. -/
def events (address : Event → Nat) (value : Memory FP32Word → Event → FP32Word) :
    List Event → Heap → Heap
  | [],h=>h
  | e::es,h=>events address value es (store h (address e) (value (observe h) e))

def cells (address : Nat → Nat) (value : Memory FP32Word → Nat → FP32Word) :
    List Nat → Heap → Heap
  | [],h=>h
  | i::is,h=>cells address value is (store h (address i) (value (observe h) i))

theorem observe_events (address : Event → Nat) (value : Memory FP32Word → Event → FP32Word)
    (es : List Event) (h : Heap) :
    observe (events address value es h)=runTagged address value es (observe h) := by
  induction es generalizing h with
  | nil=>rfl
  | cons e es ih=>simp only [events,runTagged,ih,observe_store]

theorem observe_cells (address : Nat → Nat) (value : Memory FP32Word → Nat → FP32Word)
    (is : List Nat) (h : Heap) :
    observe (cells address value is h)=runStores address value is (observe h) := by
  induction is generalizing h with
  | nil=>rfl
  | cons i is ih=>simp only [cells,runStores,ih,observe_store]

def scratch (a : Ops FP32Word) (c : CompilerLayout) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (ks : List Event) (h : Heap) : Heap :=
  events (tensorAddress bases)
    (fun m e=>nativeScratchValue a c s e.1 (currentInputs a s b m) e.2) ks h

def host (a : Ops FP32Word) (s : Shape) (trees : OutputKind → SumPlan)
    (src dst : OutputKind → Nat) (hs : List Event) (h : Heap) : Heap :=
  events (tensorAddress dst)
    (fun m e=>hostValue a (trees e.1) (e.1.shards s) (src e.1) (e.1.cells s) m e.2) hs h

def dispatch (a : Ops FP32Word) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (src dst : OutputKind → Nat)
    (ks hs : List Event) (h : Heap) : Heap :=
  if HostAPI.Empty s then h else host a s trees src dst hs (scratch a c s b src ks h)

theorem observe_dispatch (a : Ops FP32Word) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (src dst : OutputKind → Nat)
    (ks hs : List Event) (h : Heap) :
    observe (dispatch a c s b trees src dst ks hs h)=
      HostNative.dispatch a c s b trees src dst ks hs (observe h) := by
  unfold dispatch HostNative.dispatch
  split_ifs
  · rfl
  · simp only [host,scratch,observe_events,compiledRun,compiledScratch,combinedHost]

def denominator (a : Ops FP32Word) (eps : FP32Word) (s : Shape) (z den : Nat) (h : Heap) : Heap :=
  cells (fun i=>den+i)
    (fun m i=>if i<s.B*s.H*s.T then a.add (m (z+i)) eps else a.zero)
    (List.range (s.B*s.H*s.T)) h

def division (a : Ops FP32Word) (s : Shape) (o den dst : Nat) (h : Heap) : Heap :=
  cells (fun i=>dst+i)
    (fun m i=>if i<s.B*s.H*s.T*s.V then a.div (m (o+i)) (m (den+i/s.V)) else a.zero)
    (List.range (s.B*s.H*s.T*s.V)) h

def forward (a : Ops FP32Word) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (src out : OutputKind → Nat) (ks hs : List Event)
    (h : Heap) (eps : FP32Word) (den dst : Nat) (both normalize : Bool) : Heap :=
  let raw:=dispatch a c s (forwardBases x) trees src out ks hs h
  if both then raw else if normalize then
    division a s (out .forward) den dst (denominator a eps s (out .normalizer) den raw)
  else raw

theorem observe_forward (a : Ops FP32Word) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (src out : OutputKind → Nat) (ks hs : List Event)
    (h : Heap) (eps : FP32Word) (den dst : Nat) (both normalize : Bool) :
    observe (forward a c s x trees src out ks hs h eps den dst both normalize)=
      (HostNative.forward a c s x trees src out ks hs (observe h) eps den dst both normalize).api.memory := by
  cases both <;> cases normalize <;>
    simp only [forward,HostNative.forward,HostNative.forwardMemory,Bool.false_eq_true,if_false,if_true,
      division,denominator,observe_cells,observe_dispatch,HostNative.divisionRun,HostNative.denominatorRun]

def backward (a : Ops FP32Word) (c : CompilerLayout) (ctx : Saved FP32Word) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (src out : OutputKind → Nat) (ks hs : List Event) (h : Heap) : Heap :=
  dispatch (savedOps a ctx) c ctx.shape (backwardBases ctx do_ dz) trees src out ks hs h

theorem observe_backward (a : Ops FP32Word) (c : CompilerLayout) (ctx : Saved FP32Word) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (src out : OutputKind → Nat) (ks hs : List Event) (h : Heap) :
    observe (backward a c ctx do_ dz trees src out ks hs h)=
      (HostNative.backward a c ctx do_ dz trees src out ks hs (observe h)).memory := by
  exact observe_dispatch (savedOps a ctx) c ctx.shape (backwardBases ctx do_ dz) trees src out ks hs h

def propagation (a : Ops FP32Word) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (h : Heap) (dst : OutputKind → Nat) : Heap :=
  events (tensorAddress dst) (fun m ev=>HostNative.cotangentValue a s normalize tree o den dy m ev)
    (_root_.VeriTile.Bench.Repairs.ParallelAttention.events s Bool.false forwardKinds) h

theorem observe_propagation (a : Ops FP32Word) (s : Shape) (normalize : Bool) (tree : Nat → Nat → SumPlan)
    (o den : Tensor) (dy : View) (h : Heap) (dst : OutputKind → Nat) :
    observe (propagation a s normalize tree o den dy h dst)=
      HostNative.propagateRun a s normalize tree o den dy (observe h) dst := by
  exact observe_events (tensorAddress dst)
    (fun m ev=>HostNative.cotangentValue a s normalize tree o den dy m ev)
    (_root_.VeriTile.Bench.Repairs.ParallelAttention.events s Bool.false forwardKinds) h

def singleBackward (a : Ops FP32Word) (c : CompilerLayout) (f : HostNative.ForwardResult FP32Word)
    (normalize : Bool) (tree : Nat → Nat → SumPlan) (dy : View) (h : Heap)
    (dst : OutputKind → Nat) (trees : OutputKind → SumPlan)
    (src out : OutputKind → Nat) (ks hs : List Event) : Heap :=
  let s:=f.api.saved.shape
  let den:=f.denominator.getD f.api.rawNormalizer
  let ready:=propagation a s normalize tree f.api.rawOutput den dy h dst
  backward a c f.api.saved (tensor s .forward (dst .forward)) (tensor s .normalizer (dst .normalizer))
    trees src out ks hs ready

theorem observe_singleBackward (a : Ops FP32Word) (c : CompilerLayout) (f : HostNative.ForwardResult FP32Word)
    (normalize : Bool) (tree : Nat → Nat → SumPlan) (dy : View) (h : Heap)
    (dst : OutputKind → Nat) (trees : OutputKind → SumPlan)
    (src out : OutputKind → Nat) (ks hs : List Event) :
    observe (singleBackward a c f normalize tree dy h dst trees src out ks hs)=
      (HostNative.singleBackward a c f normalize tree dy (observe h) dst trees src out ks hs).memory := by
  simp only [singleBackward,HostNative.singleBackward,observe_backward,observe_propagation]

/-- Every byte, including storage gaps and sub-word offsets, follows the
proved word observation. There is no byte-frame numerical assumption. -/
theorem byte_frame (h g : Heap) (q : Nat) (same : observe h (q/4)=observe g (q/4)) : h q=g q := by
  rw [←byte_observe h q,←byte_observe g q,same]

theorem output_byte (h : Heap) (a : Nat) (x : FP32Word) (equal : observe h a=x) (k : Fin 4) :
    h (4*a+k.val)=Packing.word x k := by
  have hk:=k.isLt
  rw [←byte_observe h (4*a+k.val)]
  simpa [show (4*a+k.val)/4=a by omega,show (4*a+k.val)%4=k.val by omega] using
    congrArg (fun x=>Packing.word x k) equal

#axiomsClean observe_store
#axiomsClean observe_forward
#axiomsClean observe_backward
#axiomsClean observe_singleBackward
#axiomsClean byte_frame
#axiomsClean output_byte
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelAttention.Physical

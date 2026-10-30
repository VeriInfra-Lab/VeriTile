import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeAdmission
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeCleanup
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.K8Plan
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeBody

namespace VeriTile.Bench.Repairs.BmmChunkFwd.DirectFP
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 4000000

abbrev Byte := Fin 256
abbrev Heap := Nat → Byte
abbrev Word := BitVec 32

/-- Only numerical machine operations are ports. Raw storage leaves are
identity and have no deferred correspondence; pending standalone dots cannot
reach a materialized observation. -/
structure Arithmetic where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  mma8 : K8.Primitive

/-- Structural expansion of every variable-sized source dot into fixed K8
instructions. The graph's actual carry and its initialization/live selector
are threaded through all instructions in source reduction-coordinate order. -/
def interpret (a : Arithmetic) : Expr → Word
  | .leaf _ word => word
  | .literal p d v => a.literal p d v
  | .cast p l r x => a.cast p l r (interpret a x)
  | .binary p d op x y => a.binary p d op (interpret a x) (interpret a y)
  | .pendingDot .. => 0
  | .dotAcc c p l r out opts batch m k n left right acc i =>
      K8.plan a.mma8 c batch m k n (fun j => interpret a (left j))
        (fun j => interpret a (right j)) (interpret a acc) (K8.carryLive acc) i

/-- Complete source-body binding to the fixed K8 instruction-coordinate plan.
The operands and old/new carry are the actual reached registers. No numerical
law occurs here, and every selected instruction uses the actual live carry. -/
theorem body_k8 (arith : Arithmetic) (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (ready : FPSuccess.Ready h c s) (index : Nat)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, ∃ a : Values Expr .real [c.M,c.K], ∃ b : Values Expr .real [c.K,c.N],
      ∃ carry next : Values Expr .real [c.M,c.N],
      run (engine (backend rest c).algebra) (FPSuccess.parts h c).body
        (s.setReg "k" .nat [] (fun _ => index))=some t ∧
      s.regs .real [c.M,c.N] "acc"=some carry ∧
      t.regs .real [c.M,c.K] "a"=some a ∧ t.regs .real [c.K,c.N] "b"=some b ∧
      t.regs .real [c.M,c.N] "acc"=some next ∧
      ∀ i, interpret arith (next i)=
        K8.plan arith.mma8 c [] c.M c.K c.N (fun j => interpret arith (a j))
          (fun j => interpret arith (b j)) (interpret arith (carry i)) Bool.true i := by
  obtain ⟨t,a,b,carry,ht,hc,ha',hb',hn⟩ := body_fp32 rest h c s ready index ha hb
  exact ⟨t,a,b,carry,_,ht,hc,ha',hb',hn,fun _ => rfl⟩

def load32 (memory : Heap) (base index : Nat) : Word :=
  BitVec.ofNat 32 (((List.range 4).map fun b => (memory (base+4*index+b)).val*256^b).sum)
def load64 (memory : Heap) (base index : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (((List.range 8).map fun b => (memory (base+8*index+b)).val*256^b).sum)
def byte (w : Word) (b : Nat) : Byte := ⟨w.toNat/256^b%256,Nat.mod_lt _ (by decide)⟩

def initial (p : Metadata) (a : Direct.Allocation p) (heap : Heap) : State Expr where
  mem := fun r off =>
    if r="A" then .mk .real (.leaf .fp32 (load32 heap (base p .a) off))
    else if r="B" then .mk .real (.leaf .fp32 (load32 heap (base p .b) off))
    else if r="Seq" then .mk .int (load64 heap (base p .seq) off).toInt
    else .mk .fp32 (ofFloat .fp32 (.leaf .fp32 (load32 heap a.pointer off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

def clear (h : Host.Call) (s : State Expr) : State Expr :=
  if h.causal then
    { s with mem := fun r off => if r=h.Out then
        Cleanup.run h (.mk .fp32 (ofFloat .fp32 (.leaf .fp32 0))) (s.mem r) off else s.mem r off }
  else s

def trials (rest : Algebra Expr) (h : Host.Call) : List Host.Config → State Expr → Option (State Expr)
  | [],s => some s
  | c::cs,s => do
    let next ← FPGrid.runHost (backend rest c) h c s
    trials rest h cs next

def execute (rest : Algebra Expr) (p : Metadata) (cs : List Host.Config) (chosen : Host.Config)
    (s : State Expr) : Option (State Expr) :=
  if (shape p).outputSize=0 then some s else do
    let after ← trials rest (call p) cs s
    let result ← FPGrid.runHost (backend rest chosen) (call p) chosen after
    pure (clear (call p) result)

theorem initial_materialized (p : Metadata) (a : Direct.Allocation p) (heap : Heap) : Memory (initial p a heap) := by
  intro dtype r off
  dsimp only [initial]
  split_ifs <;> cases dtype <;> simp [toFloat,ofFloat,Cell.read,defaultValue,Materialized]

theorem trials_total (rest : Algebra Expr) (p : Metadata) (cs : List Host.Config)
    (s : State Expr) (hm : Memory s) :
    ∃ t,trials rest (call p) cs s=some t ∧ Memory t ∧ FPSuccess.RegionFrame (call p).Out s t := by
  induction cs generalizing s with
  | nil => exact ⟨s,rfl,hm,fun _ _ _ => rfl⟩
  | cons c cs ih =>
    obtain ⟨mid,hmid,hmm⟩ := host_materialized rest (call p) c s hm rfl rfl
    obtain ⟨mid',hm',frame,_,_⟩ := FPGrid.successful_host (backend rest c) (call p) c s
    have same : mid'=mid := Option.some.inj (hm'.symm.trans hmid)
    subst mid'
    obtain ⟨t,ht,htm,htf⟩ := ih mid hmm
    refine ⟨t,?_,htm,fun r off hr => (htf r off hr).trans (frame r off hr)⟩
    simp [trials,hmid,ht]

theorem clear_frame (h : Host.Call) (s : State Expr) : FPSuccess.RegionFrame h.Out s (clear h s) := by
  intro r off hr
  cases causal : h.causal <;> simp [clear,causal,hr]

theorem clear_materialized (h : Host.Call) (s : State Expr) (hm : Memory s) : Memory (clear h s) := by
  intro dtype r off
  cases causal : h.causal
  · simpa [clear,causal] using hm dtype r off
  · simp only [clear,causal,↓reduceIte]
    by_cases output : r=h.Out
    · rw [if_pos output,Cleanup.run_value]
      split_ifs
      · cases dtype <;> simp [toFloat,ofFloat,Cell.read,defaultValue,Materialized]
      · exact hm dtype r off
    · rw [if_neg output]
      exact hm dtype r off

theorem execute_total (rest : Algebra Expr) (p : Metadata) (cs : List Host.Config) (chosen : Host.Config)
    (s : State Expr) (hm : Memory s) :
    ∃ t,execute rest p cs chosen s=some t ∧ Memory t ∧ FPSuccess.RegionFrame (call p).Out s t := by
  by_cases empty : (shape p).outputSize=0
  · exact ⟨s,by simp [execute,empty],hm,fun _ _ _ => rfl⟩
  · obtain ⟨after,ha,ham,haf⟩ := trials_total rest p cs s hm
    obtain ⟨last,hl,hlm⟩ := host_materialized rest (call p) chosen after ham rfl rfl
    obtain ⟨last',hl',hlf,_,_⟩ := FPGrid.successful_host (backend rest chosen) (call p) chosen after
    have same : last'=last := Option.some.inj (hl'.symm.trans hl)
    subst last'
    exact ⟨clear (call p) last,by simp [execute,empty,ha,hl],clear_materialized _ _ hlm,
      fun r off hr => ((clear_frame _ _ r off hr).trans (hlf r off hr)).trans (haf r off hr)⟩

/-- All source numerical correspondences are per primitive at this node's
actual operands. No successful call, address law, reduction result or complete
execution equality is a premise. A consumed dot is one configured dotAcc with
its original FP32 carry and explicit TF32 K8 instruction schedule, never a whole-dot equality. -/
def Numerics (actual reference : Arithmetic) : Expr → Prop
  | .leaf _ _ => True
  | .literal p d v => actual.literal p d v=reference.literal p d v
  | .cast p l r x => Numerics actual reference x ∧
      actual.cast p l r (interpret actual x)=reference.cast p l r (interpret actual x)
  | .binary p d op x y => Numerics actual reference x ∧ Numerics actual reference y ∧
      actual.binary p d op (interpret actual x) (interpret actual y)=reference.binary p d op (interpret actual x) (interpret actual y)
  | .pendingDot .. => False
  | .dotAcc c p l r out opts batch m k n a b acc i =>
      (∀ j,Numerics actual reference (a j)) ∧ (∀ j,Numerics actual reference (b j)) ∧
      Numerics actual reference acc ∧
      K8.Agreements actual.mma8 reference.mma8 (K8.instruction c) (K8.coordinate c batch i)
        (K8.left batch m k n (fun j => interpret actual (a j)) i)
        (K8.right batch m k n (fun j => interpret actual (b j)) i)
        (interpret actual acc) (K8.carryLive acc) ((k+7)/8)


theorem numerics_eq (actual reference : Arithmetic) (x : Expr) (h : Numerics actual reference x) :
    interpret actual x=interpret reference x := by
  induction x with
  | leaf d w => rfl
  | literal p d v => exact h
  | cast p l r x ih =>
    change actual.cast p l r (interpret actual x)=_
    rw [h.2,ih h.1]; rfl
  | binary p d op x y ihx ihy =>
    change actual.binary p d op (interpret actual x) (interpret actual y)=_
    rw [h.2.2,ihx h.1,ihy h.2.1]; rfl
  | pendingDot => exact h.elim
  | dotAcc c p l r out opts batch m k n a b acc i iha ihb ihacc =>
    change K8.plan actual.mma8 c batch m k n _ _ _ _ i=_
    unfold K8.plan
    rw [K8.transport actual.mma8 reference.mma8 _ _ _ _ _ _ _ h.2.2.2]
    have ha : (fun j => interpret actual (a j))=(fun j => interpret reference (a j)) := funext (fun j => iha j (h.1 j))
    have hb : (fun j => interpret actual (b j))=(fun j => interpret reference (b j)) := funext (fun j => ihb j (h.2.1 j))
    rw [ha,hb,ihacc h.2.2.1]; rfl

def graph (p : Metadata) (s : State Expr) (off : Nat) : Expr := toFloat .fp32 ((s.mem (call p).Out off).read .fp32)

def outputHeap (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic)
    (s : State Expr) (heap : Heap) (address : Nat) : Byte :=
  if a.pointer≤address ∧ address<a.pointer+4*(shape p).outputSize then
    byte (interpret actual (graph p s ((address-a.pointer)/4))) ((address-a.pointer)%4)
  else heap address

structure Result where
  object : Nat
  shape : List Nat
  device : VeriTile.Torch.Device
  dtype : VeriTile.Torch.DType
  pointer : Nat
  heap : Heap

def result (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic) (s : State Expr) (heap : Heap) : Result :=
  ⟨a.object,(shape p).outputShape,(p.tensor .a).device,.floating .fp32,a.pointer,outputHeap p a actual s heap⟩

def runWrapper (rest : Algebra Expr) (actual : Arithmetic) (p : Metadata) (a : Direct.Allocation p)
    (cs : List Host.Config) (chosen : Host.Config) (heap : Heap) : Except InputError (Option Result) := do
  checkInputs p
  pure ((execute rest p cs chosen (initial p a heap)).map fun s => result p a actual s heap)

theorem output_byte (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic)
    (s : State Expr) (heap : Heap) (off : Nat) (ho : off<(shape p).outputSize) (b : Fin 4) :
    outputHeap p a actual s heap (a.pointer+4*off+b.val)=byte (interpret actual (graph p s off)) b.val := by
  have bound := b.isLt
  have inside : a.pointer≤a.pointer+4*off+b.val ∧ a.pointer+4*off+b.val<a.pointer+4*(shape p).outputSize := by omega
  rw [outputHeap,if_pos inside]
  have diff : a.pointer+4*off+b.val-a.pointer=4*off+b.val := by omega
  rw [diff]
  have div : (4*off+b.val)/4=off := by omega
  have mod : (4*off+b.val)%4=b.val := by omega
  rw [div,mod]

theorem output_frame (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic)
    (s : State Expr) (heap : Heap) (address : Nat)
    (outside : ¬(a.pointer≤address ∧ address<a.pointer+4*(shape p).outputSize)) :
    outputHeap p a actual s heap address=heap address := by simp [outputHeap,outside]

theorem clear_upper (p : Metadata) (s : State Expr) (causal : p.causal=Bool.true)
    (off : Nat) (upper : Cleanup.Upper (call p) off) :
    graph p (clear (call p) s) off=.leaf .fp32 0 := by
  simp only [graph,clear,show (call p).causal=Bool.true from causal,↓reduceIte]
  rw [Cleanup.run_value,if_pos upper]
  rfl

theorem execute_upper (rest : Algebra Expr) (p : Metadata) (cs : List Host.Config)
    (chosen : Host.Config) (s t : State Expr) (run : execute rest p cs chosen s=some t)
    (causal : p.causal=Bool.true) (off : Nat) (upper : Cleanup.Upper (call p) off) :
    graph p t off=.leaf .fp32 0 := by
  have positive : 0<(shape p).outputSize := by
    obtain ⟨pid,i,j,valid,eq⟩ := (Cleanup.member (call p) off).mp ((Cleanup.image (call p) off).mpr upper)
    have bound := Cleanup.offset_bound (call p) pid i j valid
    change Cleanup.offset (call p) pid i j<(shape p).outputSize at bound
    omega
  simp only [execute,if_neg (by omega : (shape p).outputSize≠0)] at run
  cases ht : trials rest (call p) cs s with
  | none => simp [ht] at run
  | some after =>
    cases hl : FPGrid.runHost (backend rest chosen) (call p) chosen after with
    | none => simp [ht,hl] at run
    | some last =>
      simp [ht,hl] at run
      subst t
      exact clear_upper p last causal off upper

/-- The complete source graph is constructed by total execution. Its choice
does not depend on the numerical operations used to interpret graph nodes. -/
def finalState (rest : Algebra Expr) (p : Metadata) (a : Direct.Allocation p)
    (cs : List Host.Config) (chosen : Host.Config) (heap : Heap) : State Expr :=
  Classical.choose (execute_total rest p cs chosen (initial p a heap) (initial_materialized p a heap))

theorem finalState_runs (rest : Algebra Expr) (p : Metadata) (a : Direct.Allocation p)
    (cs : List Host.Config) (chosen : Host.Config) (heap : Heap) :
    execute rest p cs chosen (initial p a heap)=some (finalState rest p a cs chosen heap) ∧
      Memory (finalState rest p a cs chosen heap) ∧
      FPSuccess.RegionFrame (call p).Out (initial p a heap) (finalState rest p a cs chosen heap) :=
  Classical.choose_spec (execute_total rest p cs chosen (initial p a heap) (initial_materialized p a heap))

def TaskNumerics (actual reference : Arithmetic) (rest : Algebra Expr) (p : Metadata)
    (a : Direct.Allocation p) (cs : List Host.Config) (chosen : Host.Config) (heap : Heap) : Prop :=
  ∀ off,off<(shape p).outputSize →
    Numerics actual reference (graph p (finalState rest p a cs chosen heap) off)

#axiomsClean body_k8
#axiomsClean execute_total
#axiomsClean numerics_eq
#axiomsClean output_byte
#axiomsClean output_frame
#axiomsClean execute_upper
#axiomsClean finalState_runs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.DirectFP

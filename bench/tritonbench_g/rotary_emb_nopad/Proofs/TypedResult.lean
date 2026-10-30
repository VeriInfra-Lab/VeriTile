import bench.tritonbench_g.rotary_emb_nopad.Proofs.TypedGeometry
import bench.tritonbench_g.rotary_emb_nopad.Proofs.TypedMemory
import bench.tritonbench_g.rotary_emb_nopad.Spec

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem u_other {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.UncachedLayout)
    (s data : State A) (key : Bool) (entry : State A) (r : RegionName) (o : Nat)
    (other : r≠uRegion p key) : (uBranch ops p s data key entry).mem r o=entry.mem r o :=
  Pair.frame _ _ _ _ _ _ _ (Or.inl other)

theorem u_value_afterQ {A : Type} [Inhabited A] (ops : Ops A) (r : Request)
    (s : State A) (odd : Bool) (i : ULane (uLayout r)) :
    uValue ops (uLayout r) s (uBranch ops (uLayout r) s s Bool.false s) Bool.true odd i=
      uValue ops (uLayout r) s s Bool.true odd i := by
  simp only [uValue]
  simp [u_other,uRegion,uLayout]

/-- Independent entry-memory split-half rotation. K is written only by GQA
leaders; the complete state retains every cell outside these two regions. -/
def uTarget {A : Type} [Inhabited A] (ops : Ops A) (r : Request) (s : State A) : State A :=
  let q := uBranch ops (uLayout r) s s Bool.false s
  if s.pids 0%r.groups=0 then uBranch ops (uLayout r) s s Bool.true q else q

theorem u_reference_target {A : Type} [Inhabited A] (ops : Ops A) (r : Request) (s : State A) :
    uReference ops (uLayout r) s=uTarget ops r s := by
  by_cases leader : s.pids 0%r.groups=0
  · simp only [uReference,uTarget,uLayout,leader,↓reduceIte]
    have values : uValue ops (uLayout r) s (uBranch ops (uLayout r) s s Bool.false s) Bool.true=
        uValue ops (uLayout r) s s Bool.true :=
      funext fun odd => funext (u_value_afterQ ops r s odd)
    change Pair.written _ _ _ (uValue ops (uLayout r) s (uBranch ops (uLayout r) s s Bool.false s) Bool.true) _=
      Pair.written _ _ _ (uValue ops (uLayout r) s s Bool.true) _
    rw [values]
  · simp [uReference,uTarget,uLayout,leader]

def UOutside {A : Type} (r : Request) (s : State A) (region : RegionName) (o : Nat) : Prop :=
  ∀ key,Pair.Outside (uRegion (uLayout r) key) (uActive (uLayout r) s key)
    (uOffset (uLayout r) s key) region o

theorem u_frame {A : Type} [Inhabited A] (ops : Ops A) (r : Request) (s : State A)
    (region : RegionName) (o : Nat) (outside : UOutside r s region o) :
    (uTarget ops r s).mem region o=s.mem region o := by
  unfold uTarget
  split
  · unfold uBranch
    rw [Pair.frame _ _ _ _ _ _ _ (outside Bool.true),Pair.frame _ _ _ _ _ _ _ (outside Bool.false)]
  · exact Pair.frame _ _ _ _ _ _ _ (outside Bool.false)

def ULive {A : Type} (r : Request) (s : State A) (key : Bool) (i : ULane (uLayout r)) : Prop :=
  uActive (uLayout r) s key i ∧ (key=Bool.true → s.pids 0%r.groups=0)

theorem u_readback_raw {A : Type} [Inhabited A] (ops : Ops A) (r : Request)
    (heads : ∀ key,0<r.heads key) (s : State A)
    (key odd : Bool) (i : ULane (uLayout r)) (live : ULive r s key i) :
    (uTarget ops r s).mem (uRegion (uLayout r) key)
      (uOffset (uLayout r) s key odd i)=uValue ops (uLayout r) s s key odd i := by
  cases key
  · unfold uTarget
    split
    · rw [u_other _ _ _ _ _ _ _ _ (by simp [uRegion,uLayout])]
      exact Pair.readback _ _ _ _ _ (u_ownership_raw r heads s Bool.false) odd i live.1
    · exact Pair.readback _ _ _ _ _ (u_ownership_raw r heads s Bool.false) odd i live.1
  · simp only [uTarget,live.2 rfl,↓reduceIte]
    exact Pair.readback _ _ _ _ _ (u_ownership_raw r heads s Bool.true) odd i live.1

theorem u_readback {A : Type} [Inhabited A] (ops : Ops A) (c : Checked.Call)
    (index : Indices) (accepted : Checked.check c index=Bool.true) (s : State A)
    (key odd : Bool) (i : ULane (uLayout c.request)) (live : ULive c.request s key i) :
    (uTarget ops c.request s).mem (uRegion (uLayout c.request) key)
      (uOffset (uLayout c.request) s key odd i)=uValue ops (uLayout c.request) s s key odd i := by
  apply u_readback_raw ops c.request _ s key odd i live
  intro key
  cases key
  · exact (Checked.shapes c index accepted).1
  · exact (Checked.shapes c index accepted).2.1

def COutside {A : Type} [Inhabited A] (r : Request) (s : State A) (region : RegionName) (o : Nat) : Prop :=
  ∀ key,Pair.Outside (cRegion (cLayout r) key) (cActive (cLayout r) s key)
    (cOffset (cLayout r) s key) region o

theorem c_frame {A : Type} [Inhabited A] (ops : Ops A) (r : Request) (s : State A)
    (region : RegionName) (o : Nat) (outside : COutside r s region o) :
    (cReference ops (cLayout r) s).mem region o=s.mem region o := by
  unfold cReference
  split
  · unfold cBranch
    rw [Pair.frame _ _ _ _ _ _ _ (outside Bool.false),Pair.frame _ _ _ _ _ _ _ (outside Bool.true)]
  · rfl

theorem c_readback {A : Type} [Inhabited A] (ops : Ops A) (r : Request) (s : State A)
    (past : 0≤cPast (cLayout r) s) (block : 0≤cBlock (cLayout r) s)
    (head : s.pids 0<r.qHeads) (key odd : Bool) (i : CLane (cLayout r))
    (live : cActive (cLayout r) s key i) :
    (cReference ops (cLayout r) s).mem (cRegion (cLayout r) key)
      (cOffset (cLayout r) s key odd i)=cValue ops (cLayout r) s key odd i := by
  have head' : s.pids 0<(cLayout r).Q_HEAD_NUM := head
  simp only [cReference,head',↓reduceIte]
  cases key
  · exact Pair.readback _ _ _ _ _ (c_ownership r s past block Bool.false) odd i live
  · unfold cBranch
    rw [Pair.frame _ _ _ _ _ _ _ (Or.inl (by simp [cRegion,cLayout]))]
    exact Pair.readback _ _ _ _ _ (c_ownership r s past block Bool.true) odd i live

theorem uncached_raw_successful {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (r : Request) (heads : ∀ key,0<r.heads key) (s : State A) :
    ∃ t,exec (algebra rest ops) (Kernels.uncached (uLayout r)) s=some t ∧
      t.mem=(uTarget ops r s).mem ∧
      (∀ key odd i,ULive r s key i →
        t.mem (uRegion (uLayout r) key) (uOffset (uLayout r) s key odd i)=
          uValue ops (uLayout r) s s key odd i) ∧
      (∀ region o,UOutside r s region o → t.mem region o=s.mem region o) := by
  refine ⟨(uncachedExecution rest ops (uLayout r) s).val,
    (uncachedExecution rest ops (uLayout r) s).property,?_,?_,?_⟩
  all_goals rw [uncached_memory,u_reference_target]
  · exact u_readback_raw ops r heads s
  · exact u_frame ops r s

theorem uncached_successful {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (c : Checked.Call) (index : Indices) (accepted : Checked.check c index=Bool.true) (s : State A) :
    ∃ t,exec (algebra rest ops) (Kernels.uncached (uLayout c.request)) s=some t ∧
      t.mem=(uTarget ops c.request s).mem ∧
      (∀ key odd i,ULive c.request s key i →
        t.mem (uRegion (uLayout c.request) key) (uOffset (uLayout c.request) s key odd i)=
          uValue ops (uLayout c.request) s s key odd i) ∧
      (∀ r o,UOutside c.request s r o → t.mem r o=s.mem r o) := by
  apply uncached_raw_successful rest ops c.request _ s
  intro key
  cases key
  · exact (Checked.shapes c index accepted).1
  · exact (Checked.shapes c index accepted).2.1

theorem cached_raw_successful {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (r : Request) (s : State A) (token : s.pids 1<r.tokens)
    (past : 0≤cPast (cLayout r) s) (block : 0≤cBlock (cLayout r) s) :
    ∃ t,exec (algebra rest ops) (Kernels.cached (cLayout r)) s=some t ∧
      t.mem=(cReference ops (cLayout r) s).mem ∧
      (∀ key odd i,s.pids 0<r.qHeads → cActive (cLayout r) s key i →
        t.mem (cRegion (cLayout r) key) (cOffset (cLayout r) s key odd i)=
          cValue ops (cLayout r) s key odd i) ∧
      (∀ region o,COutside r s region o → t.mem region o=s.mem region o) ∧
      (∀ o,t.mem "K" o=s.mem "K" o) := by
  refine ⟨(cachedExecution rest ops (cLayout r) s).val,
    (cachedExecution rest ops (cLayout r) s).property,?_,?_,?_,?_⟩
  all_goals rw [cached_memory rest ops _ _ token]
  · intro key odd i head live
    exact c_readback ops r s past block head key odd i live
  · exact c_frame ops r s
  · intro o
    apply c_frame
    intro key
    exact Or.inl (by cases key <;> simp [cRegion,cLayout])

theorem cached_successful {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (c : Checked.Call) (index : Indices) (accepted : Checked.check c index=Bool.true)
    (cached : c.request.cached=Bool.true) (s : State A) (observed : MetadataMatches c.request index s)
    (token : s.pids 1<c.request.tokens) (nonempty : 0<c.request.half) :
    ∃ t,exec (algebra rest ops) (Kernels.cached (cLayout c.request)) s=some t ∧
      t.mem=(cReference ops (cLayout c.request) s).mem ∧
      (∀ key odd i,s.pids 0<c.request.qHeads → cActive (cLayout c.request) s key i →
        t.mem (cRegion (cLayout c.request) key) (cOffset (cLayout c.request) s key odd i)=
          cValue ops (cLayout c.request) s key odd i) ∧
      (∀ r o,COutside c.request s r o → t.mem r o=s.mem r o) ∧
      (∀ o,t.mem "K" o=s.mem "K" o) := by
  have h := metadata_nonnegative c index accepted cached s observed token nonempty
  exact cached_raw_successful rest ops c.request s token h.1 h.2

#specNonCircular uTarget avoiding [Kernels.uncached,uncachedExecution,uReference]
#specNonCircular cReference avoiding [Kernels.cached,cachedExecution]
#axiomsClean uncached_successful
#axiomsClean cached_successful
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed

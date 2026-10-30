import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPStream
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.HostOutcome

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPWrapper
open VeriTile.Triton FP.Structural FPValues Geometry FPStorage Host
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def readValue (h : Call) (cell : Cell α) : α :=
  toFloat h.inputDType (cell.read h.inputDType.toTileDType)

def reference (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (positive : 0<c.BT)
    (logical : Nat → Nat → Nat → α) (row t s : Nat) : Cell α :=
  FPMemory.stored M h (FPExecution.expected M dot h c positive (s/BS) (logical row) t
    ⟨s%BS,Nat.mod_lt _ (by decide)⟩)

def grids (models : Config → Algebra α) (dots : Config → DotBackend α) (h : Call) (p : Storage.Placement h) :=
  fun c => FPStream.grid (models c) (dots c) h p c

theorem expected_logical (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (heap : FPStorage.Heap α) (logical : Nat → Nat → Nat → α)
    (coords : ∀ row t s,row<h.B*h.H → t<h.T → s<h.S →
      readValue h (heap (Storage.address p.backing.input (Geometry.address h row t s)))=logical row t s)
    (row : Row h) (t s : Nat) :
    FPStorage.expected M dot h p c positive heap row.val t s=reference M dot h c positive logical row.val t s := by
  unfold FPStorage.expected localExpected reference
  apply congrArg (FPMemory.stored M h)
  apply FPReference.expected_congr
  intro time feature ht hs
  exact coords row.val time feature row.isLt ht hs

theorem outcome (models : Config → Algebra α) (dots : Config → DotBackend α) (api : API (Cell α)) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (logical : Nat → Nat → Nat → α)
    (caller : FPStorage.Heap α) (live : Nat → Prop)
    (laws : PreparationLaws api h p (readValue h) logical caller live)
    (admitted : Bool) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations) :
    HostOutcome.Result h p live caller (Host.run api (grids models dots h p) h admitted tried selected caller) := by
  apply HostOutcome.outcome api h p (readValue h) logical caller live laws _ _ admitted tried valid selected chosen
  intro c hc s
  obtain ⟨t,he,_,_,frame⟩ := FPStream.grid_complete (models c) (dots c) h distinct p c (Kernels.configurations_positive c hc).1 s
  exact ⟨t,he,frame⟩

theorem successful (models : Config → Algebra α) (dots : Config → DotBackend α) (api : API (Cell α)) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (logical : Nat → Nat → Nat → α)
    (caller : FPStorage.Heap α) (live : Nat → Prop)
    (laws : PreparationLaws api h p (readValue h) logical caller live)
    (admitted : Bool) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations) (a : Allocated (Cell α))
    (prepared : prepare api admitted caller=.ok a)
    (setup : h.B*h.H*(h.T*h.S)≠0 → ReversedCumsumScalar.Host.Setups (launchAPI api) 0 (tried++[selected])) :
    ∃ t, Host.run api (grids models dots h p) h admitted tried selected caller=.ok {a with heap:=t} ∧
      metadata h p a ∧ HostOutcome.Good h p live caller t ∧
      (∀ row : Row h,∀ time feature,time<h.T → feature<h.S →
        t (Storage.address p.backing.output (Geometry.address h row.val time feature))=reference (models selected) (dots selected) h selected (Kernels.configurations_positive selected chosen).1 logical row.val time feature) ∧
      (∀ addr,¬Storage.writes h p (Storage.programs h) addr → t addr=a.heap addr) := by
  obtain ⟨metaOk,coords,_⟩ := prepare_correct api h p (readValue h) logical caller live laws admitted a prepared
  have out := outcome models dots api h distinct p logical caller live laws admitted tried valid selected chosen
  by_cases empty : h.B*h.H*(h.T*h.S)=0
  · have ret := empty_return api (grids models dots h p) h admitted tried selected caller a prepared empty
    rw [ret] at out
    refine ⟨a.heap,ret,metaOk,out.2,?_,fun _ _ => rfl⟩
    intro row time feature ht hs
    have positive : 0<h.B*h.H*(h.T*h.S) := Nat.mul_pos (Nat.zero_lt_of_lt row.isLt)
      (Nat.mul_pos (Nat.zero_lt_of_lt ht) (Nat.zero_lt_of_lt hs))
    omega
  · obtain ⟨t,he,_,outputs,frame⟩ := FPStream.tuned_complete models dots h distinct p tried valid selected chosen a.heap
    have ret := successful_return api (grids models dots h p) h admitted tried selected caller a prepared empty t he (setup empty)
    rw [ret] at out
    refine ⟨t,ret,metaOk,out.2,?_,frame⟩
    intro row time feature ht hs
    exact (outputs row time feature ht hs).trans (expected_logical (models selected) (dots selected) h p selected _ a.heap logical coords row time feature)

theorem returned (models : Config → Algebra α) (dots : Config → DotBackend α) (api : API (Cell α)) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (logical : Nat → Nat → Nat → α)
    (caller : FPStorage.Heap α) (live : Nat → Prop)
    (laws : PreparationLaws api h p (readValue h) logical caller live)
    (admitted : Bool) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations) (a result : Allocated (Cell α))
    (prepared : prepare api admitted caller=.ok a)
    (ret : Host.run api (grids models dots h p) h admitted tried selected caller=.ok result) :
    result.identity=a.identity ∧ metadata h p result ∧ HostOutcome.Good h p live caller result.heap ∧
      (∀ row : Row h,∀ time feature,time<h.T → feature<h.S →
        result.heap (Storage.address p.backing.output (Geometry.address h row.val time feature))=reference (models selected) (dots selected) h selected (Kernels.configurations_positive selected chosen).1 logical row.val time feature) ∧
      (∀ addr,¬Storage.writes h p (Storage.programs h) addr → result.heap addr=a.heap addr) := by
  have general := outcome models dots api h distinct p logical caller live laws admitted tried valid selected chosen
  rw [ret] at general
  obtain ⟨metaOk,coords,_⟩ := prepare_correct api h p (readValue h) logical caller live laws admitted a prepared
  by_cases empty : h.B*h.H*(h.T*h.S)=0
  · have same : a=result := by simpa [Host.run,prepared,empty] using ret
    subst result
    refine ⟨rfl,metaOk,general.2,?_,fun _ _ => rfl⟩
    intro row time feature ht hs
    have positive : 0<h.B*h.H*(h.T*h.S) := Nat.mul_pos (Nat.zero_lt_of_lt row.isLt)
      (Nat.mul_pos (Nat.zero_lt_of_lt ht) (Nat.zero_lt_of_lt hs))
    omega
  · cases le : ReversedCumsumScalar.Host.launches (launchAPI api) (grids models dots h p) 0 (tried++[selected]) a.heap with
    | error e => simp [Host.run,prepared,empty,le] at ret
    | ok final =>
      have same : {a with heap:=final}=result := by simpa [Host.run,prepared,empty,le] using ret
      subst result
      have raw := ReversedCumsumScalar.Host.launches_success_raw (launchAPI api) (grids models dots h p) 0 _ a.heap final le
      obtain ⟨t,he,_,outputs,frame⟩ := FPStream.tuned_complete models dots h distinct p tried valid selected chosen a.heap
      have sameHeap : t=final := Option.some.inj (he.symm.trans raw)
      subst t
      refine ⟨rfl,metaOk,general.2,?_,frame⟩
      intro row time feature ht hs
      exact (outputs row time feature ht hs).trans (expected_logical (models selected) (dots selected) h p selected _ a.heap logical coords row time feature)

#print axioms outcome
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPWrapper

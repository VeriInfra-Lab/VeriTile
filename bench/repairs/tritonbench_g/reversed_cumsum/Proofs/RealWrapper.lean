import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealStream
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.HostOutcome

namespace VeriTile.Bench.Repairs.ReversedCumsum.RealWrapper
open VeriTile.Triton Geometry RealStorage Host
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

def reference (h : Call) (logical : Nat → Nat → Nat → ℝ) (row t s : Nat) : MemCell :=
  MemCell.real (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffixValue (fun k => logical row k s) h.T t)

theorem expected_logical (h : Call) (p : Storage.Placement h) (heap : RealStorage.Heap)
    (logical : Nat → Nat → Nat → ℝ)
    (coords : ∀ row t s,row<h.B*h.H → t<h.T → s<h.S →
      read (heap (Storage.address p.backing.input (Geometry.address h row t s)))=logical row t s)
    (row : Row h) (t s : Nat) (ht : t<h.T) (hs : s<h.S) :
    RealStorage.expected h p heap row.val t s=reference h logical row.val t s := by
  unfold RealStorage.expected reference
  apply congrArg MemCell.real
  unfold VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffixValue
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.initialSum
  congr 1
  · apply Finset.sum_congr rfl
    intro k hk
    exact coords row.val k s row.isLt (Finset.mem_range.mp hk) hs
  · apply Finset.sum_congr rfl
    intro k hk
    exact coords row.val k s row.isLt ((Finset.mem_range.mp hk).trans ht) hs

theorem outcome (api : API MemCell) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (logical : Nat → Nat → Nat → ℝ)
    (caller : RealStorage.Heap) (live : Nat → Prop)
    (laws : PreparationLaws api h p read logical caller live)
    (admitted : Bool) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations) :
    HostOutcome.Result h p live caller (Host.run api (RealStream.grid h p) h admitted tried selected caller) := by
  apply HostOutcome.outcome api h p read logical caller live laws _ _ admitted tried valid selected chosen
  intro c hc s
  obtain ⟨t,he,_,_,frame⟩ := RealStream.grid_complete h distinct p c (Kernels.configurations_positive c hc).1 s
  exact ⟨t,he,frame⟩

theorem successful (api : API MemCell) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (logical : Nat → Nat → Nat → ℝ)
    (caller : RealStorage.Heap) (live : Nat → Prop)
    (laws : PreparationLaws api h p read logical caller live)
    (admitted : Bool) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations) (a : Allocated MemCell)
    (prepared : prepare api admitted caller=.ok a)
    (setup : h.B*h.H*(h.T*h.S)≠0 → ReversedCumsumScalar.Host.Setups (launchAPI api) 0 (tried++[selected])) :
    ∃ t, Host.run api (RealStream.grid h p) h admitted tried selected caller=.ok {a with heap:=t} ∧
      metadata h p a ∧ HostOutcome.Good h p live caller t ∧
      (∀ row : Row h,∀ time feature,time<h.T → feature<h.S →
        t (Storage.address p.backing.output (Geometry.address h row.val time feature))=reference h logical row.val time feature) ∧
      (∀ addr,¬Storage.writes h p (Storage.programs h) addr → t addr=a.heap addr) := by
  obtain ⟨metaOk,coords,_⟩ := prepare_correct api h p read logical caller live laws admitted a prepared
  have out := outcome api h distinct p logical caller live laws admitted tried valid selected chosen
  by_cases empty : h.B*h.H*(h.T*h.S)=0
  · have ret := empty_return api (RealStream.grid h p) h admitted tried selected caller a prepared empty
    rw [ret] at out
    refine ⟨a.heap,ret,metaOk,out.2,?_,fun _ _ => rfl⟩
    intro row time feature ht hs
    have positive : 0<h.B*h.H*(h.T*h.S) := Nat.mul_pos (Nat.zero_lt_of_lt row.isLt)
      (Nat.mul_pos (Nat.zero_lt_of_lt ht) (Nat.zero_lt_of_lt hs))
    omega
  · obtain ⟨t,he,_,outputs,frame⟩ := RealStream.tuned_complete h distinct p tried valid selected chosen a.heap
    have ret := successful_return api (RealStream.grid h p) h admitted tried selected caller a prepared empty t he (setup empty)
    rw [ret] at out
    refine ⟨t,ret,metaOk,out.2,?_,frame⟩
    intro row time feature ht hs
    exact (outputs row time feature ht hs).trans (expected_logical h p a.heap logical coords row time feature ht hs)

theorem returned (api : API MemCell) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (logical : Nat → Nat → Nat → ℝ)
    (caller : RealStorage.Heap) (live : Nat → Prop)
    (laws : PreparationLaws api h p read logical caller live)
    (admitted : Bool) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations) (a result : Allocated MemCell)
    (prepared : prepare api admitted caller=.ok a)
    (ret : Host.run api (RealStream.grid h p) h admitted tried selected caller=.ok result) :
    result.identity=a.identity ∧ metadata h p result ∧ HostOutcome.Good h p live caller result.heap ∧
      (∀ row : Row h,∀ time feature,time<h.T → feature<h.S →
        result.heap (Storage.address p.backing.output (Geometry.address h row.val time feature))=reference h logical row.val time feature) ∧
      (∀ addr,¬Storage.writes h p (Storage.programs h) addr → result.heap addr=a.heap addr) := by
  have general := outcome api h distinct p logical caller live laws admitted tried valid selected chosen
  rw [ret] at general
  obtain ⟨metaOk,coords,_⟩ := prepare_correct api h p read logical caller live laws admitted a prepared
  by_cases empty : h.B*h.H*(h.T*h.S)=0
  · have same : a=result := by simpa [Host.run,prepared,empty] using ret
    subst result
    refine ⟨rfl,metaOk,general.2,?_,fun _ _ => rfl⟩
    intro row time feature ht hs
    have positive : 0<h.B*h.H*(h.T*h.S) := Nat.mul_pos (Nat.zero_lt_of_lt row.isLt)
      (Nat.mul_pos (Nat.zero_lt_of_lt ht) (Nat.zero_lt_of_lt hs))
    omega
  · cases le : ReversedCumsumScalar.Host.launches (launchAPI api) (RealStream.grid h p) 0 (tried++[selected]) a.heap with
    | error e => simp [Host.run,prepared,empty,le] at ret
    | ok final =>
      have same : {a with heap:=final}=result := by simpa [Host.run,prepared,empty,le] using ret
      subst result
      have raw := ReversedCumsumScalar.Host.launches_success_raw (launchAPI api) (RealStream.grid h p) 0 _ a.heap final le
      obtain ⟨t,he,_,outputs,frame⟩ := RealStream.tuned_complete h distinct p tried valid selected chosen a.heap
      have sameHeap : t=final := Option.some.inj (he.symm.trans raw)
      subst t
      refine ⟨rfl,metaOk,general.2,?_,frame⟩
      intro row time feature ht hs
      exact (outputs row time feature ht hs).trans (expected_logical h p a.heap logical coords row time feature ht hs)

#print axioms outcome
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealWrapper

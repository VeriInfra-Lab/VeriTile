import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.CheckedBridge
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.LaunchScale

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section

/-- Independent prefix-sum formula, including the source's finite sentinel
and guarded retention of an earlier normalized accumulator. Its inputs are
arbitrary Real arrays, without an encoding or FP finiteness restriction. -/
def mathematical (a : Host.Arguments) (u : Inputs) (l : Launch)
    (i : RealIO.Lane (Host.layout a)) : ℝ :=
  Public.normalizedBlocks (Host.layout a) l u (LaunchScale.value a.q.dims) i.1.val i.2.1.val
    (count (Host.layout a) l u)

theorem whole_mathematical_real (a : Host.Arguments) (r : Regions) (v : Views)
    (A : FlatAlloc) (s : BlockState) (requestedMax : Nat)
    (checked : checkInputs a r v A.extent (inputs s r) requestedMax=Bool.true)
    (labels : Labels r) (launchScalar : LaunchScale.Bound a) (disjoint : A.Disjoint) (covered : ∀ rg,rg∉A.regions → A.extent rg=0)
    (zero : s.undef=(fun _ _ => 0)) (owner : Nat) :
    Spec.Real (∃ final,
      RealStream.run a r (HostRun.allPrograms a) s=some final ∧
      RealStream.flatRun a r A (HostRun.allPrograms a) (A.flattenState s)=some (A.flattenState final) ∧
      (∀ idx i,Host.active a r s idx i →
        final.readMem r.out (RealStream.offset a r s idx i)=
          mathematical a (inputs s r) (Host.launchOf a idx) i) ∧
      (∀ (b h row d : Nat),b < a.batch → h < a.q.heads →
        (row : Int) < (inputs s r).sequence b-(inputs s r).prompt b → d < a.q.dims →
        ∃ idx : GridIndex (Host.grid a),∃ i : RealIO.Lane (Host.layout a),
          Host.active a r s idx i ∧ (Host.launchOf a idx).batch=b ∧
          (Host.launchOf a idx).head=h ∧ queryRow (Host.layout a) (Host.launchOf a idx) i.1.val=row ∧ i.2.1.val=d) ∧
      (∀ rg off,¬RealStream.writes a r s (HostRun.allPrograms a) rg off → final.mem rg off=s.mem rg off) ∧
      (∀ rg∈readRegions r,∀ off,final.mem rg off=s.mem rg off) ∧
      (HostRun.finish a owner final).pythonReturn=none ∧
      (HostRun.finish a owner final).outputOwner=owner ∧
      (HostRun.finish a owner final).descriptor=a.o) := by
  change Host.scale a=LaunchScale.value a.q.dims at launchScalar
  have guard := (checkInputs_iff a r v A.extent (inputs s r) requestedMax).mp checked
  have hv := valid a r A s guard.2.1 guard.2.2.1 labels
  obtain ⟨final,run,flat,same,undef,values,frame⟩ := RealStream.complete a r A s hv disjoint covered zero (HostRun.allPrograms a)
  refine ⟨final,run,flat,?_,Host.covers a r A s hv,frame,?_,rfl,rfl,rfl⟩
  · intro idx i hi
    have value := values idx (by simp [HostRun.allPrograms]) i hi
    have result := value.trans (Public.expected_eq _ _ _ _ _).symm
    simpa only [Public.expected,mathematical,launchScalar] using result
  · intro rg read off
    apply frame
    rintro ⟨idx,_,i,_,eq,_⟩
    exact labels rg read eq.symm

#specNonCircular mathematical avoiding [VeriTile.Triton.exec,RealStream.run,RealStream.flatRun,kernel]
#axiomsClean whole_mathematical_real
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked

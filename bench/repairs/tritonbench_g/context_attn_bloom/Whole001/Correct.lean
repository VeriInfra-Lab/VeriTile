import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.Host

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Public
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt l.batch then
    scale * ∑ e : Fin p.D, query p l a i e.val * key p l a j e.val
  else -100000000

def denominator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN), if j.val < c*p.BN then Real.exp (score p l a scale i j.val) else 0

def numerator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val)*value p l a j.val d else 0

/-- Independent prefix-sum recurrence. The separate acc_scale guard leaves
the earlier accumulator unscaled for wholly future blocks. Replacing this
formula with ideal causal softmax would change the source's real behavior. -/
def normalizedBlocks (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) : Nat → ℝ
  | 0 => 0
  | c+1 => normalizedBlocks p l a scale i d c *
      (if ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
        denominator p l a scale i c / denominator p l a scale i (c+1) else 1) +
      (numerator p l a scale i d (c+1)-numerator p l a scale i d c) / denominator p l a scale i (c+1)

def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : RealIO.Lane p) : ℝ :=
  normalizedBlocks p l a scale i.1.val i.2.1.val (count p l a)

theorem expected_eq (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : RealIO.Lane p) :
    expected p scale l a i=RealIO.expected p scale l a i := by
  exact RealIO.recurrence_eq p l a scale i.1.val i.2.1.val
    (normalizedBlocks p l a scale i.1.val i.2.1.val) rfl (fun _ => rfl) _

/-- Whole caller invocation over logical tensor inputs. Only local tensor API
contracts, layout/metadata/storage conditions and concrete index bounds are
premises. The successful current-memory source run, flat run, independent
formula, caller frame, supplied output identity and None return are conclusions.
The finite -1e8 mask and the separate accumulator guard are intentional. -/
specification host_real (a : Host.Arguments) (r : Regions) (d : Preparation.Data)
    (api : Preparation.LocalAPI r d) (initial : Preparation.Runtime) (owner : Nat)
    (layout : HostRun.LayoutDomain a) (A : FlatAlloc)
    (valid : Host.Admissible a r A (Preparation.prepare api initial).real)
    (disjoint : A.Disjoint) (allocations : ∀ rg, rg∉A.regions → A.extent rg=0)
    (zero : initial.real.undef=(fun _ _ => 0)) (base : RegionName → Nat)
    (abi : ABI.IndexABI a r d.inputs base)
    (jobs : List (GridIndex (Host.grid a))) (schedule : HostRun.CompleteSchedule a jobs) :
    Spec.Real (ABI.SourceExact a r d.inputs base ∧ ∃ final,
      HostRun.real a r d api initial owner jobs=some (HostRun.finish a owner final) ∧
      RealStream.flatRun a r A jobs (A.flattenState (Preparation.prepare api initial).real)=
        some (A.flattenState final) ∧
      (∀ idx i, Host.active a r (Preparation.prepare api initial).real idx i →
        final.readMem r.out (RealStream.offset a r (Preparation.prepare api initial).real idx i)=
          expected (Host.layout a) (Host.scale a) (Host.launchOf a idx) d.inputs i) ∧
      (∀ rg off, api.visible rg →
        ¬RealStream.writes a r (Preparation.prepare api initial).real jobs rg off →
        final.mem rg off=initial.real.mem rg off) ∧
      (HostRun.finish a owner final).pythonReturn=none ∧
      (HostRun.finish a owner final).outputOwner=owner ∧
      (HostRun.finish a owner final).descriptor=a.o) := by
  have prepared := Preparation.prepared_inputs (a:=a) r d api initial
  obtain ⟨final,hr,hflat,hin,hundef,hvalues,hframe⟩ := RealStream.complete a r A
    (Preparation.prepare api initial).real valid disjoint allocations
    ((Preparation.undef_frame api initial).trans zero) jobs
  refine ⟨ABI.source_exact a r d.inputs base abi,final,?_,hflat,?_,?_,rfl,rfl,rfl⟩
  · simp only [HostRun.real,hr,Option.map_some]
  · intro idx i hi
    have value := hvalues idx (HostRun.covered a jobs schedule idx) i hi
    unfold Host.result at value
    rw [prepared.1] at value
    exact value.trans (expected_eq _ _ _ _ _).symm
  · intro rg off visible outside
    exact (hframe rg off outside).trans (Preparation.caller_frame api initial rg visible off).1

#axiomsClean host_real
#axiomsClean expected_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Public

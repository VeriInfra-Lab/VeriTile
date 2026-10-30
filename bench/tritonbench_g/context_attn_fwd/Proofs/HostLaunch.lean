import bench.tritonbench_g.context_attn_fwd.Proofs.HostInterface

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 1000000

def flatSource (a : Arguments) (r : Regions) (A : FlatAlloc) (swapped : Bool) : Kernel :=
  A.flattenKernel (source a r swapped).eraseDType.toAlgKernel

def active (a : Arguments) (r : Regions) (s : BlockState)
    (idx : GridIndex (grid a)) (i : RealIO.Lane (layout a)) : Prop :=
  RealIO.active (layout a) (launchOf a idx) (inputs s r) i

def address (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (idx : GridIndex (grid a)) (i : RealIO.Lane (layout a)) : MemCellAddr :=
  (A.flat, A.addr r.out (RealIO.write (layout a) (launchOf a idx) (inputs s r) i))

/-- Independent finite-sentinel weighted sum over the original tensor views. -/
def result (a : Arguments) (r : Regions) (s : BlockState)
    (idx : GridIndex (grid a)) (i : RealIO.Lane (layout a)) : ℝ :=
  RealIO.closedExpected (layout a) (scale a) (launchOf a idx) (inputs s r) i

def writes (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState) : WriteFootprint :=
  HostComposition.allWrites (grid a) (active a r s) (address a r A s)

/-- Python updates the supplied O allocation and has no returned tensor. -/
def returned (final : BlockState) : BlockState × Option Unit := (final, none)

theorem complete (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (ha : Admissible a r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary (flatSource a r A swapped)
        (grid a) (A.flattenState s) final) ∧
      (∀ idx i, active a r s idx i →
        final.readMem (address a r A s idx i).1 (address a r A s idx i).2 = result a r s idx i) ∧
      BlockState.WriteWithin (writes a r A s) (A.flattenState s) final ∧
      (returned final).2 = none := by
  have per : ∀ idx, ∃ final,
      exec (flatSource a r A swapped) ((A.flattenState s).withGridIndex idx) = some final ∧
      (∀ i, active a r s idx i →
        final.readMem (address a r A s idx i).1 (address a r A s idx i).2 = result a r s idx i) ∧
      BlockState.WriteWithin
        (HostComposition.writes (grid a) (active a r s) (address a r A s) idx)
        ((A.flattenState s).withGridIndex idx) final := by
    intro idx
    have valid : RealIO.valid (layout a) (launchOf a idx) (inputs s r) := by
      intro i j hi hj he
      exact (unique_writer a r A s ha idx idx i j hi hj he).2
    obtain ⟨v, he, ho, hf⟩ := RealIO.implements_closed (layout a) r .fp32 swapped (scale a)
      (block_positive a) A hd hcov (s.withGridIndex idx)
      (by simpa [RealIO.sourceIO] using valid)
      (by simpa [RealIO.sourceIO] using ha.bounds idx)
    refine ⟨v, he, ho, ?_⟩
    intro rr oo hn
    apply hf rr oo
    intro hw
    apply hn
    rcases hw with ⟨hflat, rg, off, ⟨i, hi, hr, hoff⟩, heq⟩
    refine ⟨i, hi, ?_⟩
    change r.out = rg at hr
    change RealIO.write (layout a) (launchOf a idx) (inputs s r) i = off at hoff
    subst rg
    subst off
    exact Prod.ext hflat.symm heq
  have owned : ∀ idx jdx i j, active a r s idx i → active a r s jdx j →
      address a r A s idx i = address a r A s jdx j → idx = jdx := by
    intro idx jdx i j hi hj he
    have hh := congrArg Prod.snd he
    have he' : RealIO.write (layout a) (launchOf a idx) (inputs s r) i =
        RealIO.write (layout a) (launchOf a jdx) (inputs s r) j := Nat.add_left_cancel hh
    exact (unique_writer a r A s ha idx jdx i j hi hj he').1
  obtain ⟨final, hl, ho, hf⟩ := HostComposition.assemble (grid a)
    (flatSource a r A swapped) (A.flattenState s) (active a r s) (address a r A s)
    (result a r s) per owned
  exact ⟨final, hl, ho, hf, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host

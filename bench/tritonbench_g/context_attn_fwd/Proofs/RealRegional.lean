import bench.tritonbench_g.context_attn_fwd.IO
import bench.tritonbench_g.context_attn_fwd.Proofs.Mathematical_HostComposition

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealRegional
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section

/-- Whole ordinary grid assembly from the existing per-source theorem.
Every program reads the same initial arrays; no serial scheduling is used. -/
theorem complete (p : Layout) (r : Regions) (g : Grid) (scale : ℝ) (s : BlockState)
    (positive : 0 < p.BN)
    (unique : ∀ e f : RealInterface.Event p g,
      RealInterface.active p g (Program.inputs s r) e →
      RealInterface.active p g (Program.inputs s r) f →
      RealInterface.address p g (Program.inputs s r) e=RealInterface.address p g (Program.inputs s r) f → e=f) :
    ∃ final,Nonempty (Kernel.GridLaunchedOrdinary (RealInterface.source p r scale).kernel.toAlgKernel g s final) ∧
      (∀ e : RealInterface.Event p g,RealInterface.active p g (Program.inputs s r) e →
        final.readMem r.out (RealInterface.address p g (Program.inputs s r) e)=
          RealIO.closedExpected p scale (RealInterface.launch e.1) (Program.inputs s r) e.2) ∧
      (∀ rg off,(∀ e : RealInterface.Event p g,RealInterface.active p g (Program.inputs s r) e →
        rg≠r.out ∨ off≠RealInterface.address p g (Program.inputs s r) e) → final.mem rg off=s.mem rg off) := by
  have programs : ∀ idx : GridIndex g,∃ final,
      exec (RealInterface.source p r scale).kernel.toAlgKernel (s.withGridIndex idx)=some final ∧
      (∀ i,RealInterface.active p g (Program.inputs s r) (idx,i) →
        final.readMem r.out (RealInterface.address p g (Program.inputs s r) (idx,i))=
          RealIO.closedExpected p scale (RealInterface.launch idx) (Program.inputs s r) i) ∧
      BlockState.WriteWithin
        (HostComposition.writes g (fun idx i => RealInterface.active p g (Program.inputs s r) (idx,i))
          (fun idx i => (r.out,RealInterface.address p g (Program.inputs s r) (idx,i))) idx)
        (s.withGridIndex idx) final := by
    intro idx
    obtain ⟨final,ran,values,frame⟩ := EpilogueResult.correct p r .fp32 Bool.false scale
      (s.withGridIndex idx) positive (fun i j hi hj he => congrArg Prod.snd (unique (idx,i) (idx,j) hi hj he))
    refine ⟨final,ran,?_,?_⟩
    · intro i hi
      have result := values i hi
      change _=RealIO.expected p scale (RealInterface.launch idx) (Program.inputs s r) i at result
      rw [RealIO.expected_closed] at result
      exact result
    · intro rg off outside
      symm
      apply frame
      rintro ⟨i,hi,hr,ho⟩
      exact outside ⟨i,hi,Prod.ext hr ho⟩
  obtain ⟨final,ran,values,frame⟩ := HostComposition.assemble g _ s
    (fun idx i => RealInterface.active p g (Program.inputs s r) (idx,i))
    (fun idx i => (r.out,RealInterface.address p g (Program.inputs s r) (idx,i)))
    (fun idx i => RealIO.closedExpected p scale (RealInterface.launch idx) (Program.inputs s r) i)
    programs (fun idx jdx i j hi hj he => congrArg Prod.fst
      (unique (idx,i) (jdx,j) hi hj (congrArg Prod.snd he)))
  refine ⟨final,ran,fun e he => values e.1 e.2 he,?_⟩
  intro rg off outside
  symm
  apply frame
  rintro ⟨idx,i,hi,he⟩
  have bad := outside (idx,i) hi
  exact bad.elim (fun hr => hr (congrArg Prod.fst he).symm)
    (fun ho => ho (congrArg Prod.snd he).symm)

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealRegional

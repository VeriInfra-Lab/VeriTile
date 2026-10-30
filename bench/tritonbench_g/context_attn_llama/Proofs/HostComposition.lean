import bench.tritonbench_g.context_attn_llama.Proofs.RealIO
import VeriTile.Triton.Launch.Composition

/-! Finite launch composition. The executions below are constructed from the
existing source theorem; neither successful execution nor output values are
premises. This helper is also used by the Mistral host adapter. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.HostComposition
open VeriTile.Triton
noncomputable section

variable {ι : Type} (g : Grid) (A : FlatAlloc) (s : BlockState) (k : Kernel)
  (active : GridIndex g → ι → Prop) (out : RegionName)
  (offset : GridIndex g → ι → Nat) (expected : GridIndex g → ι → ℝ)

def writes (idx : GridIndex g) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ i, active idx i ∧ A.addr out (offset idx i) = cell.2

def footprint : WriteFootprint := fun cell => ∃ idx, writes g A active out offset idx cell

theorem finite_grid
    (runs : ∀ idx : GridIndex g, ∃ final,
      exec k ((A.flattenState s).withGridIndex idx) = some final ∧
      (∀ i, active idx i → final.readMem A.flat (A.addr out (offset idx i)) = expected idx i) ∧
      BlockState.WriteWithin (writes g A active out offset idx)
        ((A.flattenState s).withGridIndex idx) final)
    (owner : ∀ idx jdx i j, active idx i → active jdx j →
      offset idx i = offset jdx j → idx = jdx) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary k g (A.flattenState s) final) ∧
      (∀ idx i, active idx i → final.readMem A.flat (A.addr out (offset idx i)) = expected idx i) ∧
      BlockState.WriteWithin (footprint g A active out offset) (A.flattenState s) final := by
  classical
  let fs : Kernel.GridFrames k g (A.flattenState s) := fun idx =>
    { final := Classical.choose (runs idx)
      writes := writes g A active out offset idx
      h_exec := (Classical.choose_spec (runs idx)).1
      h_writeWithin := (Classical.choose_spec (runs idx)).2.2 }
  have hd : Kernel.GridWritesDisjoint fs := by
    intro idx jdx hn cell hi hj
    obtain ⟨_,i,hai,hei⟩ := hi
    obtain ⟨_,j,haj,hej⟩ := hj
    apply hn (owner idx jdx i j hai haj _)
    have he := hei.trans hej.symm
    exact Nat.add_left_cancel he
  refine ⟨Kernel.mergeFrames g (A.flattenState s) fs, ⟨⟨fs,hd,rfl⟩⟩,?_,?_⟩
  · intro idx i hi
    have hm := Kernel.mergeFrames_mem_written hd idx A.flat (A.addr out (offset idx i))
      (show (fs idx).writes (A.flat,A.addr out (offset idx i)) from ⟨rfl,i,hi,rfl⟩)
    have hr : (Kernel.mergeFrames g (A.flattenState s) fs).readMem A.flat
        (A.addr out (offset idx i)) = (fs idx).final.readMem A.flat (A.addr out (offset idx i)) := by
      simp only [BlockState.readMem,hm]
    rw [hr]
    exact (Classical.choose_spec (runs idx)).2.1 i hi
  · exact Kernel.mergeFrames_writeWithin fs

/-- A full-cell frame preserves each bounded input allocation disjoint from
the output allocation. This establishes input stability across all writers. -/
theorem input_stable
    (hd : A.Disjoint) (hc : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (hb : ∀ idx i, active idx i → offset idx i < A.extent out)
    (final : BlockState)
    (hf : BlockState.WriteWithin (footprint g A active out offset) (A.flattenState s) final)
    (rg : RegionName) (hne : out ≠ rg) (o : Nat) (ho : o < A.extent rg) :
    final.mem A.flat (A.addr rg o) = (A.flattenState s).mem A.flat (A.addr rg o) := by
  apply (hf _ _ _).symm
  rintro ⟨idx,_,i,hi,he⟩
  have hr : rg ∈ A.regions := by
    by_contra hn
    rw [hc rg hn] at ho
    omega
  have hout : out ∈ A.regions := by
    by_contra hn
    have h := hb idx i hi
    rw [hc out hn] at h
    omega
  have hbound := hb idx i hi
  have heq := congrArg A.decode he
  rw [A.decode_addr hd hout hbound,A.decode_addr hd hr ho] at heq
  exact hne (congrArg Prod.fst (Option.some.inj heq))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.HostComposition

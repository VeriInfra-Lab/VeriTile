import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.IO
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
open Classical
set_option maxHeartbeats 2000000
variable {α : Type}

structure Agreement [Inhabited α] (c : Call) (r : Regions) (memory : RegionName → Nat → α)
    (s : State α) : Prop where
  metadata : FPHostSource.Metadata c r s
  words : ∀ region offset,ReadHit c r region offset → (s.mem region offset).read .real=memory region offset

theorem input_agreement [Inhabited α] (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (xs : (io c r o).Inputs) (hd : domain c r xs) (s : State α) (hin : (io c r o).InputBinding xs s) :
    Agreement c r (initialMemory c r xs) s := by
  constructor
  · constructor
    · intro b hb
      exact (hin 3 (⟨b,hb⟩,PUnit.unit) trivial).trans (hd.1 ⟨b,hb⟩)
    · intro b hb
      exact (hin 4 (⟨b,hb⟩,PUnit.unit) trivial).trans (hd.2 ⟨b,hb⟩)
  · intro region offset hit
    unfold initialMemory
    rw [dif_pos hit]
    have selected := Classical.choose_spec hit
    dsimp only
    split_ifs with hq hk
    · exact (congrArg₂ (fun rg off => (s.mem rg off).read .real) hq.1 hq.2).trans
        (hin 0 _ selected.1)
    · exact (congrArg₂ (fun rg off => (s.mem rg off).read .real) hk.1 hk.2).trans
        (hin 1 _ selected.1)
    · have hv := selected.2.resolve_left hq
      have hv := hv.resolve_left hk
      exact (congrArg₂ (fun rg off => (s.mem rg off).read .real) hv.1 hv.2).trans
        (hin 2 _ selected.1)

@[simp] theorem job_initial (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (s : State α) : (job c r o idx).initial s=FPHost.atProgram c idx s := rfl

#axiomsClean input_agreement
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData

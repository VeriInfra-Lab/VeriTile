import bench.tritonbench_g.context_attn_nopad.Proofs.IOLogicalGrid
import bench.tritonbench_g.context_attn_nopad.Proofs.IOSchedule
import bench.tritonbench_g.context_attn_nopad.Proofs.IOOutputGeometry

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

def expected (c : Call) (r : Regions) (o : FPBackend.Ops α) (xs : ChannelValues α (inputs c r)) : ChannelValues α (outputs c r) :=
  fun | ⟨0,_⟩ => fun j =>
    IOLogical.grid c r o (FPHost.jobs c) (initialMemory c r xs) r.out (address c c.o j)

theorem complete (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence) :
    (io c r o).ImplementsOn (domain c r) (expected c r o) := by
  intro xs hd bounds hb initial hin
  have agree := input_agreement c r o xs hd initial hin
  refine ⟨schedule_safe c r o xs bounds hb startLabel lengthLabel (FPHost.jobs c) initial agree.metadata,?_⟩
  obtain ⟨result,ran,hm⟩ := FPReference.whole c r o (FPHost.jobs c) initial
  have graph := IOLogical.grid_agreement c r o (FPHost.jobs c) initial (initialMemory c r xs) agree startLabel lengthLabel
  refine ⟨result,(schedule_run c r o (FPHost.jobs c) initial).trans ran,?_,?_⟩
  · intro out j on
    fin_cases out
    change mapCell id (result.mem r.out (address c c.o j))=Cell.mk .real _
    rw [hm,graph.2 r.out (address c c.o j) (Or.inr (output_covered c r j on))]
    rfl
  · intro region off outside
    rw [hm]
    exact (FPReference.grid_frame c r o (FPHost.jobs c) initial agree.metadata startLabel lengthLabel).2
      region off (outside_written c r o xs region off outside)

#axiomsClean complete
#specNonCircular expected avoiding [FPReference.grid,FPReference.runGrid,FP.Structural.exec,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData

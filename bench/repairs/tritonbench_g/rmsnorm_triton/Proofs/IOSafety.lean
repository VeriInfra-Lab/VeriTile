import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.IOGeometry
import bench.tritonbench_g.rmsnorm_triton.Proofs.Safety
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton
open Whole KernelGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem active_injective (g : Geometry) (unique : OutputUnique g.parameters)
    (b m : Nat) (hb : b<g.batches) (hm : m<g.rows) (s : BlockState) :
    Function.Injective (fun k : Fin g.width => Full.Wb.outOff ((launch b m).initial s)
      (g.output.stride 0) (g.output.stride 1) (g.output.stride 2) k.val) := by
  intro k k' same
  apply Fin.ext
  exact (unique b m k.val b m k'.val hb hm k.isLt hb hm k'.isLt
    (by simpa [outputOffset,Geometry.parameters,Full.Wb.outOff,Nat.add_assoc] using
      congrArg (g.output.base+·) same)).2.2

 theorem source_safe (g : Geometry) (unique : OutputUnique g.parameters)
    (bounds : RegionBounds) (s : BlockState) (vals : (io g).Ctx)
    (bounded : (io g).RegionalBounds bounds s vals)
    (b m : Nat) (hb : b<g.batches) (hm : m<g.rows) (t : BlockState) :
    (launch b m).Safe (io g).kernel.toAlgKernel bounds t := by
  let q := index g b m hb hm
  have pos : 0<rowCount g := Nat.lt_of_le_of_lt (Nat.zero_le q.val) q.isLt
  have hx (k : Fin g.width) :
      b*g.input.stride 0+m*g.input.stride 1+k.val*g.input.stride 2<bounds "x" := by
    simpa [io,q,inputAddress] using bounded.1 q.castSucc k pos
  have hw (k : Fin g.width) : k.val*g.weights.stride 0<bounds "weights" := by
    simpa [io] using bounded.1 (Fin.last (rowCount g)) k pos
  have ho (k : Fin g.width) :
      b*g.output.stride 0+m*g.output.stride 1+k.val*g.output.stride 2<bounds "out" := by
    simpa [io,q,outputAddress] using bounded.2.1 q k trivial
  change Kernel.TraceSafeR .triv bounds (io g).kernel.toAlgKernel ((launch b m).initial t)
  dsimp only [io]
  rw [kernel_source]
  apply (Safety.typed_safe_iff bounds .fp32 .fp32 .fp32 Bool.false _ _ _ _ _ _ _ _ _ _ _ _ _ _).mpr
  apply rmsnorm_traceSafeR_of_injective .triv bounds _ _ _ _ _ _ _ _ _ _ _ _ _
    (by decide : 0<blockSize) (by decide) (by decide) _ (active_injective g unique b m hb hm t)
  · intro step j active
    simpa using hx ⟨step.val*blockSize+j.val,active⟩
  · intro step j active
    exact hw ⟨step.val*blockSize+j.val,active⟩
  · intro step j active
    simpa using ho ⟨step.val*blockSize+j.val,active⟩

 theorem schedule_safe (g : Geometry) (unique : OutputUnique g.parameters)
    (work : Schedule g.parameters) (bounds : RegionBounds) (s : BlockState)
    (vals : (io g).Ctx) (bounded : (io g).RegionalBounds bounds s vals) :
    AlgorithmScheduleSafe (io g).kernel.toAlgKernel bounds (schedule g work vals) s := by
  apply algorithmScheduleSafe_of_forall
  intro job member t
  change job ∈ work.order.map (fun bm => launch bm.1 bm.2) at member
  obtain ⟨⟨b,m⟩,hmem,heq⟩ := List.mem_map.mp member
  subst job
  have h := member_bounds g work b m hmem
  exact source_safe g unique bounds s vals bounded b m h.1 h.2 t

#axiomsClean source_safe
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.RmsnormTriton.RmsIO

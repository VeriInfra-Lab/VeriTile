import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.IOSafety
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.Source
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.ActiveOutput

namespace VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton
open Whole KernelGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem row_input (g : Geometry) (s : BlockState) (vals : (io g).Ctx)
    (inputs : (io g).RegionalInputs s vals) (q : Fin (rowCount g)) (k : Fin g.width) :
    s.readMem "x" (inputAddress g q.val k.val)=vals q.castSucc k := by
  have pos : 0<rowCount g := Nat.lt_of_le_of_lt (Nat.zero_le q.val) q.isLt
  simpa [io,UKernelIO.RegionalInputs,q.isLt] using inputs q.castSucc k pos

theorem weight_input (g : Geometry) (s : BlockState) (vals : (io g).Ctx)
    (inputs : (io g).RegionalInputs s vals) (q : Fin (rowCount g)) (k : Fin g.width) :
    s.readMem "weights" (k.val*g.weights.stride 0)=vals (Fin.last (rowCount g)) k := by
  have pos : 0<rowCount g := Nat.lt_of_le_of_lt (Nat.zero_le q.val) q.isLt
  simpa [io] using inputs (Fin.last (rowCount g)) k pos

theorem row_run (g : Geometry) (unique : OutputUnique g.parameters)
    (s : BlockState) (vals : (io g).Ctx) (inputs : (io g).RegionalInputs s vals)
    (b m : Nat) (hb : b<g.batches) (hm : m<g.rows) :
    ∃ final, (launch b m).run (io g).kernel.toAlgKernel s=some final ∧
      (∀ k : Fin g.width, final.readMem "out"
        (outputAddress g (index g b m hb hm).val k.val)=
          target g vals (index g b m hb hm) k) ∧
      (∀ (r : RegionName) address, (r≠"out" ∨ ∀ k : Fin g.width,
        address≠outputAddress g (index g b m hb hm).val k.val) →
        final.mem r address=s.mem r address) := by
  let q := index g b m hb hm
  obtain ⟨final,ran,values,frame⟩ := ActiveOutput.run .triv "x" "weights" "out"
    (g.input.stride 0) (g.input.stride 1) (g.input.stride 2) (g.weights.stride 0)
    (g.output.stride 0) (g.output.stride 1) (g.output.stride 2) g.width blockSize g.eps
    (by decide) (by decide) (by decide) ((launch b m).initial s)
    (active_injective g unique b m hb hm s)
  have hx (j : Fin g.width) :
      s.readMem "x" (b*g.input.stride 0+m*g.input.stride 1+j.val*g.input.stride 2)=
        vals q.castSucc j := by
    simpa [q,inputAddress] using row_input g s vals inputs q j
  have hw (j : Fin g.width) := weight_input g s vals inputs q j
  refine ⟨final,?_,?_,?_⟩
  · change exec (io g).kernel.toAlgKernel ((launch b m).initial s)=some final
    dsimp only [io]
    rw [Source.original_run]
    simpa only [execR_triv] using ran
  · intro k
    have hread (r : RegionName) (a : Nat) :
        ((launch b m).initial s).readMem r a=s.readMem r a := rfl
    have v := values k
    simp only [Full.Wb.outOff,launch_pid0,launch_pid1,Full.Final2.rmsSpecFull,
      Full.Final2.meanSq,Full.ScratchRms.xOff,hread,pow_two,hx,hw,
      WithBot.realRsqrt_some,WithBot.unbotD_some,one_div] at v
    simpa only [outputAddress,batch_index,row_index,target,rms] using v
  · intro r address outside
    apply frame r address
    simpa [Full.Wb.outOff,outputAddress] using outside

#axiomsClean row_run
end
end VeriTile.Bench.Repairs.RmsnormTriton.RmsIO

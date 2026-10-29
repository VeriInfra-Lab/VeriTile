import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RealImplementation
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxHeartbeats 2000000

def snapshot (c : Call) (s : BlockState) : Inputs c := fun i j =>
  s.readMem (inputRegion i) (inputAddress c i ((inputCoordinates c i).symm j))

theorem snapshot_binding (c : Call) (s : BlockState) :
    (programIO c).InputBinding s (snapshot c s) := by
  intro i j _
  rfl

theorem localInputs_snapshot (c : Call) (s : BlockState) (idx : GridIndex c.grid) :
    localInputs c (snapshot c s) idx=Whole.inputs c s idx := by
  simp [localInputs,snapshot,inputAddress,inputRegion,Whole.inputs,Call.io,RealContract.io,
    RealIO.IO.inputWindow,Call.params,Geometry.Strides.at,Call.carryAddress,Call.outAddress,Call.row,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem formula_snapshot (c : Call) (s : BlockState) (o : Fin 3) (j : OutputPoint c o) :
    formula c (snapshot c s) o (outputCoordinates c o j)=
      Whole.result c s j.1 (outputLane c o j.2) := by
  simp only [formula,Equiv.symm_apply_apply,localInputs_snapshot]
  fin_cases o <;> rfl

/-- Exact previously proved canonical flattening transport, now explicitly
connected to the finite mathematical input/output ports. No arbitrary-flat-state
extension, new separation condition, or new source execution proof is assumed. -/
theorem canonical_flat_io (c : Call) (valid : Valid c) (A : FlatAlloc)
    (allocated : Allocated c A.extent) (disjoint : A.Disjoint)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (s : BlockState) (defined : s.undef=(fun _ _=>0)) :
    ∃ final,Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid (A.flattenState s) final) ∧
      (∀ o (j : OutputPoint c o),outputActive c o →
        final.readMem A.flat (A.addr (outputRegion o) (outputAddress c o j))=
          formula c (snapshot c s) o (outputCoordinates c o j)) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState s) final := by
  obtain ⟨final,ran,result,frame⟩ := RawKernel.whole_real c valid A allocated disjoint closed s defined
  refine ⟨final,ran,?_,frame⟩
  intro o j active
  rw [formula_snapshot]
  have live : Ownership.active c (outputLane c o j.2) := by fin_cases o <;> exact active
  have value := result j.1 (outputLane c o j.2) live
  simpa only [Flat.address,output_location] using value

#axiomsClean canonical_flat_io
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

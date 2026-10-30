import bench.tritonbench_g.attention_score.Proofs.Reused.NativeColumn
import bench.tritonbench_g.attention_score.Proofs.Reused.RealPhysical

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeReturnedMath
open VeriTile Triton FP.Structural NativeInstructions NativeStaging NativeIdeal Geometry Physical
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem inputs_eq (decode : BitVec 32 → ℝ) (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    NativeMath.inputs decode c.D (parameters c offset size scale window complement)
      (FPGrid.programState c (Physical.initial abi (NativePhysical.input physical)) (FPReturned.ownerJob c z h j))=
      DenseSource.inputs c (RealPhysical.initial abi (fun a => decode (physical a))) z.val h.val := by
  obtain ⟨hz,hh⟩ := DenseSource.batch_head c z.val h
  unfold NativeMath.inputs DenseSource.inputs RealLoads.inputs
  congr 1
  all_goals
    simp [NativeMath.inputs,FPContext.baseQ,FPContext.baseK,FPContext.baseM,FPGrid.programState,
      FPReturned.ownerJob,Physical.initial,FPStorage.state,FPStorage.importMemory,FPStorage.cell,
      FPStorage.project,NativePhysical.input,lower,Graph.eval,isa,DenseSource.inputs,RealLoads.inputs,
      RealPhysical.initial,BlockState.readMem,parameters,hz,hh,
      Shape.queryBase,Shape.keyBase,Shape.normalizerBase,
      Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals
    have hz' : (h.val+z.val*c.H)/c.H=z.val := by simpa [Nat.add_comm] using hz
    simp [hz',Nat.mod_eq_of_lt h.isLt,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

/-- The complete native graph's exact-real interpretation is the same
independent logical sum proved for the erased source. -/
theorem target_ideal (decode : BitVec 32 → ℝ) (hz : decode 0=0) (c : Shape) (abi : ABI c)
    (physical : Nat → BitVec 32) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    (NativePhysical.target c abi physical offset size scale window complement z h j).eval (isa decode)=
      RealReturned.target c (RealPhysical.initial abi (fun a => decode (physical a)))
        offset size scale window complement z h j := by
  have howner := (owner c j).2.2
  unfold NativePhysical.target NativeLoop.returnedGraph
  rw [NativeColumn.logical_column decode hz c.B c.D c.block _ _ _
    (by simpa [FPGrid.programState,FPReturned.ownerJob,FPReturned.ownerLane,parameters,howner] using j.isLt)
    (by simp [parameters])]
  rw [inputs_eq]
  simp [RealReturned.target,parameters,FPGrid.programState,FPReturned.ownerJob,FPReturned.ownerLane,howner]

theorem target_logical (decode : BitVec 32 → ℝ) (hz : decode 0=0) (c : Shape) (abi : ABI c)
    (physical : Nat → BitVec 32) (xs : LogicalInputs ℝ c)
    (normalized : Normalized c abi (fun a => decode (physical a)) xs)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    (NativePhysical.target c abi physical offset size scale window complement z h j).eval (isa decode)=
      RealPhysical.logicalTarget c xs offset size scale window complement z h j :=
  (target_ideal decode hz c abi physical offset size scale window complement z h j).trans
    (RealPhysical.target_logical c abi (fun a => decode (physical a)) xs normalized offset size scale window complement z h j)

#axiomsClean target_logical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeReturnedMath

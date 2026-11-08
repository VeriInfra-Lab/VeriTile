import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealModelIO
import bench.tritonbench_g.attention_fwd_triton3.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def modelInputEncoded (c : Call) (i : Fin 6) (j : Fin (inputArity c i)) :
    TileIndex (modelInputPorts c)[i].shape :=
  cast (congrArg TileIndex (congrArg IOChannel.shape (model_input_port c i))).symm (j,PUnit.unit)

@[simp] theorem model_index_encoded (c : Call) (i : Fin 6) (j : Fin (inputArity c i)) :
    modelInputIndex c i (modelInputEncoded c i j)=j := by fin_cases i <;> rfl

theorem model_input_mask (c : Call) (v : Inputs c) (i : Fin 6) (j : InputPoint c i)
    (active : inputActive c i j) :
    (modelCore c).inputMask i (packInputs c v)
      (modelInputEncoded c i (inputCoordinates c i j)) := by
  change inputActive c i ((inputCoordinates c i).symm _)
  simpa only [model_index_encoded,Equiv.symm_apply_apply] using active

theorem model_input_bound (c : Call) (v : Inputs c) (bounds : RegionName → Nat)
    (bounded : (modelCore c).InBounds bounds (packInputs c v)) (i : Fin 6) (j : InputPoint c i)
    (active : inputActive c i j) : inputAddress c i j<bounds (inputRegion i) := by
  have h:=bounded.2.1 i (modelInputEncoded c i (inputCoordinates c i j)) (model_input_mask c v i j active)
  fin_cases i
  · change inputAddress c 0 ((inputCoordinates c 0).symm ((inputCoordinates c 0) j))<bounds (inputRegion 0) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change inputAddress c 1 ((inputCoordinates c 1).symm ((inputCoordinates c 1) j))<bounds (inputRegion 1) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change inputAddress c 2 ((inputCoordinates c 2).symm ((inputCoordinates c 2) j))<bounds (inputRegion 2) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change inputAddress c 3 ((inputCoordinates c 3).symm ((inputCoordinates c 3) j))<bounds (inputRegion 3) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change inputAddress c 4 ((inputCoordinates c 4).symm ((inputCoordinates c 4) j))<bounds (inputRegion 4) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change inputAddress c 5 ((inputCoordinates c 5).symm ((inputCoordinates c 5) j))<bounds (inputRegion 5) at h
    rw [Equiv.symm_apply_apply] at h
    exact h

theorem model_input_pin (c : Call) (v : Inputs c) (s : State Real.R)
    (binding : (modelCore c).InputBinding (packInputs c v) s) (i : Fin 6) (j : InputPoint c i)
    (active : inputActive c i j) :
    (s.mem (inputRegion i) (inputAddress c i j)).read .real=some (v i (inputCoordinates c i j)) := by
  have h:=binding i (modelInputEncoded c i (inputCoordinates c i j)) (model_input_mask c v i j active)
  fin_cases i
  · change (s.mem (inputRegion 0) (inputAddress c 0 ((inputCoordinates c 0).symm ((inputCoordinates c 0) j)))).read .real=some (v 0 ((inputCoordinates c 0) j)) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change (s.mem (inputRegion 1) (inputAddress c 1 ((inputCoordinates c 1).symm ((inputCoordinates c 1) j)))).read .real=some (v 1 ((inputCoordinates c 1) j)) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change (s.mem (inputRegion 2) (inputAddress c 2 ((inputCoordinates c 2).symm ((inputCoordinates c 2) j)))).read .real=some (v 2 ((inputCoordinates c 2) j)) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change (s.mem (inputRegion 3) (inputAddress c 3 ((inputCoordinates c 3).symm ((inputCoordinates c 3) j)))).read .real=some (v 3 ((inputCoordinates c 3) j)) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change (s.mem (inputRegion 4) (inputAddress c 4 ((inputCoordinates c 4).symm ((inputCoordinates c 4) j)))).read .real=some (v 4 ((inputCoordinates c 4) j)) at h
    rw [Equiv.symm_apply_apply] at h
    exact h
  · change (s.mem (inputRegion 5) (inputAddress c 5 ((inputCoordinates c 5).symm ((inputCoordinates c 5) j)))).read .real=some (v 5 ((inputCoordinates c 5) j)) at h
    rw [Equiv.symm_apply_apply] at h
    exact h

theorem model_output_bound (c : Call) (v : Inputs c) (bounds : RegionName → Nat)
    (bounded : (modelCore c).InBounds bounds (packInputs c v)) (o : Fin 3) (j : OutputPoint c o)
    (active : outputActive c o) : outputAddress c o j<bounds (outputRegion o) := by
  have h:=bounded.2.2.1 o (NativeInterface.outputEncoded c o j) active
  change outputAddress c o (NativeInterface.outputPoint c o (NativeInterface.outputEncoded c o j))<
    bounds (modelOutputPorts c)[o].buf at h
  rw [NativeInterface.outputPoint_encoded] at h
  fin_cases o <;> exact h

theorem model_windows (c : Call) (v : Inputs c) (bounds : RegionName → Nat)
    (bounded : (modelCore c).InBounds bounds (packInputs c v)) (idx : GridIndex c.grid) (s : State Real.R) :
    SourceSafety.Windows c
      (Reference.inputs (Reference.Composition.prepare c.grid idx s)) bounds := by
  let a := Reference.inputs (Reference.Composition.prepare c.grid idx s)
  have pid0 : a.tile=c.tile idx := by
    simp [a,Reference.inputs,Reference.Composition.prepare,GridIndex.toPids,Call.grid,Grid.rank,Call.tile]
  have pid1 : a.hz=c.hz idx := by
    simp [a,Reference.inputs,Reference.Composition.prepare,GridIndex.toPids,Call.grid,Grid.rank,Call.hz]
  have query : SourceSafety.BlockBounds bounds (Reference.qptr c a)
      [c.BM,c.D] (if c.params.evenM≠0 then [] else [0,1]) := by
    intro j checked
    have active : c.params.evenM≠0 ∨ c.tile idx*c.BM+j.1.val<c.NQ := by
      by_cases he : c.params.evenM≠0
      · exact Or.inl he
      · right
        have h : c.tile idx*c.BM+j.1.val<c.NQ ∧ j.2.1.val<c.D := by
          simpa [Reference.qptr,pid0,he] using checked
        exact h.1
    have h := model_input_bound c v bounds bounded 0 (idx,j) active
    simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
      RealIO.IO.inputWindow,Call.params,Reference.qptr,
      Reference.batch,Reference.head,pid0,pid1,Nat.add_assoc] using h
  constructor
  · constructor
    · exact query
    · intro r
      have h := model_output_bound c v bounds bounded 0 (idx,r,PUnit.unit) trivial
      simpa [RealInterface.outputAddress,RealInterface.outputRegion,Call.io,RealContract.io,
        RealIO.IO.outputWindowFor,Call.params,Reference.carryAddress,Reference.row,
        pid0,pid1,Nat.add_assoc] using h
    · intro active r
      rcases active with hi|hf
      · have h := model_input_bound c v bounds bounded 4 (idx,r,PUnit.unit) hi
        simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
          RealIO.IO.inputWindow,Call.params,Reference.carryAddress,Reference.row,
          pid0,pid1,Nat.add_assoc] using h
      · have h := model_output_bound c v bounds bounded 1 (idx,r,PUnit.unit) hf
        simpa [RealInterface.outputAddress,RealInterface.outputRegion,Call.io,RealContract.io,
          RealIO.IO.outputWindowFor,Call.params,Reference.carryAddress,Reference.row,
          pid0,pid1,Nat.add_assoc] using h
    · intro j _
      have h := model_output_bound c v bounds bounded 2 (idx,j) trivial
      simpa [RealInterface.outputAddress,RealInterface.outputRegion,Call.io,RealContract.io,
        RealIO.IO.outputWindowFor,Call.params,Reference.optr,Reference.batch,Reference.head,
        pid0,pid1,Nat.add_assoc] using h
  · intro k d
    have h := model_input_bound c v bounds bounded 1 (idx,k,d,PUnit.unit) trivial
    simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
      RealIO.IO.inputWindow,Call.params,Reference.batch,Reference.head,Reference.kvhead,
      pid1,Nat.add_assoc] using h
  · intro k d
    have h := model_input_bound c v bounds bounded 2 (idx,k,d,PUnit.unit) trivial
    simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
      RealIO.IO.inputWindow,Call.params,Reference.batch,Reference.head,Reference.kvhead,
      pid1,Nat.add_assoc] using h

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

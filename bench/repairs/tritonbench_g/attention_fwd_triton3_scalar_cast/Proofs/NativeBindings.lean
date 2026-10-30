import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.NativeIO
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeLocality
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.SourceSafetyGeometry

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxHeartbeats 2000000

theorem reads_mem (c : Call) (i : Fin 6) (j : RealInterface.InputPoint c i)
    (active : RealInterface.inputActive c i j) : RealInterface.inputAddress c i j∈reads c i := by
  apply List.mem_map.mpr
  refine ⟨RealInterface.inputCoordinates c i j,?_,?_⟩
  · apply List.mem_filter.mpr
    exact ⟨List.mem_finRange _,by simpa only [Equiv.symm_apply_apply,decide_eq_true_eq] using active⟩
  · simp

theorem input_bound (c : Call) (v : Inputs) (bounds : RegionName → Nat)
    (bounded : (programIO c).InBounds bounds v) (i : Fin 6) (j : RealInterface.InputPoint c i)
    (active : RealInterface.inputActive c i j) :
    RealInterface.inputAddress c i j<bounds (RealInterface.inputRegion i) := by
  have h := bounded.2.1 i (RealInterface.inputAddress c i j) (reads_mem c i j active)
  fin_cases i <;> exact h

theorem output_bound (c : Call) (v : Inputs) (bounds : RegionName → Nat)
    (bounded : (programIO c).InBounds bounds v) (o : Fin 3) (j : RealInterface.OutputPoint c o)
    (active : RealInterface.outputActive c o) :
    RealInterface.outputAddress c o j<bounds (RealInterface.outputRegion o) := by
  have h := bounded.2.2.1 o (outputEncoded c o j) active
  change RealInterface.outputAddress c o (outputPoint c o (outputEncoded c o j))<
    bounds (outputPorts c)[o].buf at h
  rw [outputPoint_encoded] at h
  fin_cases o <;> exact h

theorem input_binding (c : Call) (v : Inputs) (s : State Native.Expr)
    (bound : (programIO c).InputBinding v s) : ArrayLocality.SameReads s (arrayState v) := by
  intro rg member off
  simp only [ArrayLocality.buffers,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals
    first
    | simpa [programIO,inputPorts,arrayState,arrayRead,arrayValues,mapValue] using bound (0 : Fin 6) off
    | simpa [programIO,inputPorts,arrayState,arrayRead,arrayValues,mapValue] using bound (1 : Fin 6) off
    | simpa [programIO,inputPorts,arrayState,arrayRead,arrayValues,mapValue] using bound (2 : Fin 6) off
    | simpa [programIO,inputPorts,arrayState,arrayRead,arrayValues,mapValue] using bound (3 : Fin 6) off
    | simpa [programIO,inputPorts,arrayState,arrayRead,arrayValues,mapValue] using bound (4 : Fin 6) off
    | simpa [programIO,inputPorts,arrayState,arrayRead,arrayValues,mapValue] using bound (5 : Fin 6) off

theorem windows (c : Call) (v : Inputs) (bounds : RegionName → Nat)
    (bounded : (programIO c).InBounds bounds v) (idx : GridIndex c.grid) (s : State Native.Expr) :
    SourceSafety.Windows (ScaleBinding.template c)
      (Reference.inputs (Reference.Composition.prepare c.grid idx s)) bounds := by
  let a := Reference.inputs (Reference.Composition.prepare c.grid idx s)
  have pid0 : a.tile=c.tile idx := by
    simp [a,Reference.inputs,Reference.Composition.prepare,GridIndex.toPids,Call.grid,Grid.rank,Call.tile]
  have pid1 : a.hz=c.hz idx := by
    simp [a,Reference.inputs,Reference.Composition.prepare,GridIndex.toPids,Call.grid,Grid.rank,Call.hz]
  have query : SourceSafety.BlockBounds bounds (Reference.qptr (ScaleBinding.template c) a)
      [c.BM,c.D] (if c.params.evenM≠0 then [] else [0,1]) := by
    intro j checked
    have active : c.params.evenM≠0 ∨ c.tile idx*c.BM+j.1.val<c.NQ := by
      by_cases he : c.params.evenM≠0
      · exact Or.inl he
      · right
        have h : c.tile idx*c.BM+j.1.val<c.NQ ∧ j.2.1.val<c.D := by
          simpa [Reference.qptr,ScaleBinding.template,pid0,he] using checked
        exact h.1
    have h := input_bound c v bounds bounded 0 (idx,j) active
    simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
      RealIO.IO.inputWindow,Call.params,Reference.qptr,ScaleBinding.template,
      Reference.batch,Reference.head,pid0,pid1,Nat.add_assoc] using h
  constructor
  · constructor
    · exact query
    · intro r
      have h := output_bound c v bounds bounded 0 (idx,r,PUnit.unit) trivial
      simpa [RealInterface.outputAddress,RealInterface.outputRegion,Call.io,RealContract.io,
        RealIO.IO.outputWindowFor,Call.params,Reference.carryAddress,Reference.row,
        ScaleBinding.template,pid0,pid1,Nat.add_assoc] using h
    · intro active r
      rcases active with hi|hf
      · have h := input_bound c v bounds bounded 4 (idx,r,PUnit.unit) hi
        simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
          RealIO.IO.inputWindow,Call.params,Reference.carryAddress,Reference.row,
          ScaleBinding.template,pid0,pid1,Nat.add_assoc] using h
      · have h := output_bound c v bounds bounded 1 (idx,r,PUnit.unit) hf
        simpa [RealInterface.outputAddress,RealInterface.outputRegion,Call.io,RealContract.io,
          RealIO.IO.outputWindowFor,Call.params,Reference.carryAddress,Reference.row,
          ScaleBinding.template,pid0,pid1,Nat.add_assoc] using h
    · intro j _
      have h := output_bound c v bounds bounded 2 (idx,j) trivial
      simpa [RealInterface.outputAddress,RealInterface.outputRegion,Call.io,RealContract.io,
        RealIO.IO.outputWindowFor,Call.params,Reference.optr,Reference.batch,Reference.head,
        ScaleBinding.template,pid0,pid1,Nat.add_assoc] using h
  · intro k d
    have h := input_bound c v bounds bounded 1 (idx,k,d,PUnit.unit) trivial
    simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
      RealIO.IO.inputWindow,Call.params,Reference.batch,Reference.head,Reference.kvhead,
      ScaleBinding.template,pid1,Nat.add_assoc] using h
  · intro k d
    have h := input_bound c v bounds bounded 2 (idx,k,d,PUnit.unit) trivial
    simpa [RealInterface.inputAddress,RealInterface.inputRegion,Call.io,RealContract.io,
      RealIO.IO.inputWindow,Call.params,Reference.batch,Reference.head,Reference.kvhead,
      ScaleBinding.template,pid1,Nat.add_assoc] using h

#axiomsClean input_binding
#axiomsClean windows
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface

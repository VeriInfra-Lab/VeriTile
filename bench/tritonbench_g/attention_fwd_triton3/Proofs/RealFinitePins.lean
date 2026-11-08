import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealFiniteLoads

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def PinnedAt (c : Call) (v : Inputs c) (idx : GridIndex c.grid) (s : State Real.R) : Prop :=
  ∀ (i : Fin 6) (j : TileIndex (inputShape c i)), inputActive c i (idx,j) →
    (s.mem (inputRegion i) (inputAddress c i (idx,j))).read .real=some (v i (inputCoordinates c i (idx,j)))

theorem finite_pins_at (c : Call) (v : Inputs c) (s : State Real.R)
    (idx : GridIndex c.grid) (binding : PinnedAt c v idx s) :
    Real.Pins c (Reference.inputs (Reference.Composition.prepare c.grid idx s)) (localInputs c v idx) := by
  let a:=Reference.inputs (Reference.Composition.prepare c.grid idx s)
  have pid0 : a.tile=c.tile idx := by
    simp [a,Reference.inputs,Reference.Composition.prepare,GridIndex.toPids,Call.grid,Grid.rank,Call.tile]
  have pid1 : a.hz=c.hz idx := by
    simp [a,Reference.inputs,Reference.Composition.prepare,GridIndex.toPids,Call.grid,Grid.rank,Call.hz]
  change Real.Pins c a (localInputs c v idx)
  constructor
  · intro r d active
    have live : inputActive c 0 (idx,r,d,PUnit.unit) := by simpa [inputActive,pid0] using active
    have h:=binding 0 (r,d,PUnit.unit) live
    simpa [a,Reference.inputs,Reference.Composition.prepare,localInputs,inputAddress,inputRegion,
      Call.io,RealContract.io,RealIO.IO.inputWindow,Call.params,Reference.qptr,
      Reference.batch,Reference.head,pid0,pid1,Nat.add_assoc] using h
  · intro k d
    have h:=binding 1 (k,d,PUnit.unit) trivial
    simpa [a,Reference.inputs,Reference.Composition.prepare,localInputs,inputAddress,inputRegion,
      Call.io,RealContract.io,RealIO.IO.inputWindow,Call.params,Reference.batch,Reference.head,
      Reference.kvhead,pid0,pid1,Nat.add_assoc] using h
  · intro k d
    have h:=binding 2 (k,d,PUnit.unit) trivial
    simpa [a,Reference.inputs,Reference.Composition.prepare,localInputs,inputAddress,inputRegion,
      Call.io,RealContract.io,RealIO.IO.inputWindow,Call.params,Reference.batch,Reference.head,
      Reference.kvhead,pid0,pid1,Nat.add_assoc] using h
  · intro r read
    have live : inputActive c 3 (idx,r,PUnit.unit) := read
    have h:=binding 3 (r,PUnit.unit) live
    simpa [a,Reference.inputs,Reference.Composition.prepare,localInputs,inputAddress,inputRegion,
      Call.io,RealContract.io,RealIO.IO.inputWindow,Call.params,Reference.carryAddress,Reference.row,
      pid0,pid1,Nat.add_assoc] using h
  · intro r read
    have live : inputActive c 4 (idx,r,PUnit.unit) := read
    have h:=binding 4 (r,PUnit.unit) live
    simpa [a,Reference.inputs,Reference.Composition.prepare,localInputs,inputAddress,inputRegion,
      Call.io,RealContract.io,RealIO.IO.inputWindow,Call.params,Reference.carryAddress,Reference.row,
      pid0,pid1,Nat.add_assoc] using h
  · intro r d read
    have live : inputActive c 5 (idx,r,d,PUnit.unit) := read
    have h:=binding 5 (r,d,PUnit.unit) live
    simpa [a,Reference.inputs,Reference.Composition.prepare,localInputs,inputAddress,inputRegion,
      Call.io,RealContract.io,RealIO.IO.inputWindow,Call.params,Reference.optr,
      Reference.batch,Reference.head,pid0,pid1,Nat.add_assoc] using h

theorem finite_pins (c : Call) (v : Inputs c) (s : State Real.R)
    (binding : (modelCore c).InputBinding (packInputs c v) s) (idx : GridIndex c.grid) :
    Real.Pins c (Reference.inputs (Reference.Composition.prepare c.grid idx s)) (localInputs c v idx) :=
  finite_pins_at c v s idx (fun i j live=>model_input_pin c v s binding i (idx,j) live)

#axiomsClean finite_pins
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RealBindings

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
open Host Ownership
noncomputable section
set_option maxHeartbeats 3000000

def localResult (c : Call) (v : Inputs c) (idx : GridIndex c.grid) : Lane c.BM c.D → ℝ
  | .center r => (RealContract.expected c.BM c.BN c.D c.finish (c.tile idx) c.params
      (localInputs c v idx)).center r
  | .denominator r => (RealContract.expected c.BM c.BN c.D c.finish (c.tile idx) c.params
      (localInputs c v idx)).denominator r
  | .output r d => (RealContract.expected c.BM c.BN c.D c.finish (c.tile idx) c.params
      (localInputs c v idx)).output (r,d,PUnit.unit)

theorem program (c : Call) (valid : Valid c) (v : Inputs c) (s : BlockState)
    (bound : (programIO c).InputBinding s v) (idx : GridIndex c.grid) :
    ∃ t,exec c.source (s.withGridIndex idx)=some t ∧
      (∀ lane,active c lane →
        t.readMem (address c idx lane).1 (address c idx lane).2=localResult c v idx lane) ∧
      BlockState.WriteWithin (HostComposition.writes c.grid (fun _ => active c) (address c) idx)
        (s.withGridIndex idx) t := by
  obtain ⟨t,execution,results⟩ := RealSource.executes c.BM c.BN c.D c.Z c.finish c.params Bool.false
    (s.withGridIndex idx) valid.blockN (evenN c)
    (RealOutputLayout.output_layout c.BM c.D c.params _ valid.output.dim_positive
      valid.output.row_stride (by simp [Call.params]) (by simp [Call.params]) (by simp [Call.params]))
  have result := RealContract.readback c.BM c.BN c.D c.Z c.finish c.params Bool.false
    (s.withGridIndex idx) t (localInputs c v idx)
    (RealContract.input_pins c.BM c.BN c.D c.Z c.finish c.params Bool.false _ _ (pins c v s bound idx)) results
  refine ⟨t,execution,?_,?_⟩
  · intro lane live
    cases lane with
    | center r =>
      simpa [address,localResult,Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc]
        using result.center (r,PUnit.unit)
    | denominator r =>
      simpa [address,localResult,Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc]
        using result.denominator live (r,PUnit.unit)
    | output r d => exact result.output (r,d,PUnit.unit)
  · intro rg off outside
    symm
    apply result.frame rg off
    · by_cases hm : rg="M"
      · right
        intro i heq
        apply outside
        refine ⟨.center i.1,trivial,?_⟩
        apply Prod.ext hm.symm
        simpa [Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc] using heq.symm
      · exact Or.inl hm
    · by_cases hf : c.finish=0
      · right
        by_cases hl : rg="L"
        · right
          intro i heq
          apply outside
          refine ⟨.denominator i.1,hf,?_⟩
          apply Prod.ext hl.symm
          simpa [Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc] using heq.symm
        · exact Or.inl hl
      · exact Or.inl hf
    · by_cases ho : rg="O"
      · right
        intro i heq
        apply outside
        refine ⟨.output i.1 i.2.1,trivial,?_⟩
        exact Prod.ext ho.symm heq.symm
      · exact Or.inl ho

def outputLane (c : Call) : (o : Fin 3) → TileIndex (outputShape c o) → Lane c.BM c.D
  | 0 => fun j => .center j.1
  | 1 => fun j => .denominator j.1
  | 2 => fun j => .output j.1 j.2.1

theorem output_location (c : Call) (o : Fin 3) (idx : GridIndex c.grid)
    (j : TileIndex (outputShape c o)) :
    address c idx (outputLane c o j)=(outputRegion o,outputAddress c o (idx,j)) := by
  fin_cases o <;> simp [outputLane,address,outputRegion,outputAddress,Call.io,RealContract.io,
    RealIO.IO.outputWindowFor,Call.params,Call.carryAddress,Call.row,Call.outAddress,
    Geometry.Strides.at,Nat.add_assoc]

theorem frame (c : Call) (v : Inputs c) (s t : BlockState)
    (hf : BlockState.WriteWithin (Whole.writes c) s t) : (programIO c).Frame s t v := by
  intro rg off ho _
  symm
  apply hf
  have avoid (o : Fin 3) (j : OutputPoint c o) (live : outputActive c o) :
      (outputRegion o,outputAddress c o j)≠(rg,off) := by
    intro same
    have h := ho o (outputCoordinates c o j) live
    simp only [programIO,Equiv.symm_apply_apply] at h
    exact h.elim (fun ne => ne (congrArg Prod.fst same).symm)
      (fun ne => ne (congrArg Prod.snd same).symm)
  rintro ⟨idx,lane,live,loc⟩
  cases lane with
  | center r =>
      exact avoid 0 (idx,r,PUnit.unit) trivial
        ((output_location c 0 idx (r,PUnit.unit)).symm.trans loc)
  | denominator r =>
      exact avoid 1 (idx,r,PUnit.unit) live
        ((output_location c 1 idx (r,PUnit.unit)).symm.trans loc)
  | output r d =>
      exact avoid 2 (idx,r,d,PUnit.unit) trivial
        ((output_location c 2 idx (r,d,PUnit.unit)).symm.trans loc)

theorem implements (c : Call) (valid : Valid c) : (programIO c).Implements (formula c) := by
  intro v _ before bound
  let s := AlgorithmInitialUndef.zero.initial before
  have pinned : (programIO c).InputBinding s v := bound
  obtain ⟨t,execution,outputs,outside⟩ := HostComposition.assemble c.grid c.source s
    (fun _=>active c) (address c) (localResult c v) (program c valid v s pinned)
    (fun i j x y _ _ he=>(unique c valid i j x y he).1)
  have head : (AlgorithmStage.ordinary (selectedSource c) c.grid).Runs s t := by
    simpa only [AlgorithmStage.Runs,source_projection] using execution
  let trace : AlgorithmExecution [.ordinary (selectedSource c) c.grid] s t := .cons head (.nil t)
  refine ⟨t,trace,?_,?_,frame c v before t outside⟩
  · intro bounds bounded
    change (∀ idx : GridIndex c.grid,(selectedSource c).kernel.toAlgKernel.TraceSafe bounds
      (s.withGridIndex idx)) ∧ True
    refine ⟨?_,trivial⟩
    intro idx
    rw [source_projection]
    exact safe c valid v s bounds bounded idx
  · intro o j live
    let point := (outputCoordinates c o).symm j
    have active : active c (outputLane c o point.2) := by
      fin_cases o <;> exact live
    have value := outputs point.1 (outputLane c o point.2) active
    rw [output_location] at value
    fin_cases o <;> exact value

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

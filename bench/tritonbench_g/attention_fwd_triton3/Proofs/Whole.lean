import bench.tritonbench_g.attention_fwd_triton3.Proofs.Ownership
import bench.tritonbench_g.context_attn_fwd.Proofs.Mathematical_HostComposition

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Whole
open VeriTile.Triton Geometry Host Ownership
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section
set_option maxHeartbeats 2000000

/-- Inputs are pinned solely to the initial arrays, including raw carry.
No assumption is made about a previous END call's interpretation. -/
def inputs (c : Call) (s : BlockState) (idx : GridIndex c.grid) : RealInputs.Inputs c.BM c.D c.NK where
  query := fun r d => s.readMem "Q" (c.q.at (c.hz idx/c.H) (c.hz idx%c.H) (c.row idx r.val) d.val)
  key := fun k d => s.readMem "K" (c.k.at (c.hz idx/c.H) ((c.hz idx%c.H)/(c.H/c.HK)) k.val d.val)
  value := fun k d => s.readMem "V" (c.v.at (c.hz idx/c.H) ((c.hz idx%c.H)/(c.H/c.HK)) k.val d.val)
  center := fun r => s.readMem "M" (c.carryAddress idx r.val)
  denominator := fun r => s.readMem "L" (c.carryAddress idx r.val)
  output := fun r d => s.readMem "O" (c.outAddress idx r.val d.val)

theorem pins (c : Call) (s : BlockState) (idx : GridIndex c.grid) :
    RealIO.Pins c.io (s.withGridIndex idx) (inputs c s idx) := by
  constructor <;> intros <;>
    simp [inputs,Call.io,RealContract.io,Call.params,Call.carryAddress,Call.outAddress,Call.row,
      Strides.at,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,BlockState.readMem]

/-- Complete independent six-input maximum/weighted-sum formula. Its Real
bottom/total-division conventions are inherited explicitly, not IEEE laws. -/
def expected (c : Call) (s : BlockState) (idx : GridIndex c.grid) :=
  Correctness.expected c.BM c.BN c.D c.finish (c.tile idx) c.params (inputs c s idx)

def result (c : Call) (s : BlockState) (idx : GridIndex c.grid) : Lane c.BM c.D → ℝ
  | .center r => (expected c s idx).center r
  | .denominator r => (expected c s idx).denominator r
  | .output r d => (expected c s idx).output (r,d,PUnit.unit)

def writes (c : Call) := HostComposition.allWrites c.grid (fun _ => active c) (address c)

theorem safe (c : Call) (h : Valid c) (a : Allocated c extent)
    (s : BlockState) (idx : GridIndex c.grid) : c.source.TraceSafe extent (s.withGridIndex idx) := by
  exact RealSource.safe c.BM c.BN c.D c.Z c.finish c.params Bool.false _ extent h.blockN (evenN c)
    (RealAddressBounds.trace_bounds c.BM c.BN c.D c.finish c.params _ extent h.blockN (evenN c)
      (RealContract.address_bounds c.BM c.BN c.D c.Z c.finish c.params Bool.false _ extent (bounds c h a s idx)))

theorem program (c : Call) (h : Valid c) (s : BlockState) (idx : GridIndex c.grid) :
    ∃ final, exec c.source (s.withGridIndex idx)=some final ∧
      (∀ i,active c i → final.readMem (address c idx i).1 (address c idx i).2=result c s idx i) ∧
      BlockState.WriteWithin (HostComposition.writes c.grid (fun _ => active c) (address c) idx)
        (s.withGridIndex idx) final := by
  obtain ⟨t,he,hy⟩:=RealSource.executes c.BM c.BN c.D c.Z c.finish c.params Bool.false
    (s.withGridIndex idx) h.blockN (evenN c)
    (RealOutputLayout.output_layout c.BM c.D c.params _ h.output.dim_positive h.output.row_stride
      (by simp [Call.params]) (by simp [Call.params]) (by simp [Call.params]))
  have hr:=RealContract.readback c.BM c.BN c.D c.Z c.finish c.params Bool.false
    (s.withGridIndex idx) t (inputs c s idx)
    (RealContract.input_pins c.BM c.BN c.D c.Z c.finish c.params Bool.false _ _ (pins c s idx)) hy
  refine ⟨t,he,?_,?_⟩
  · intro i hi
    cases i with
    | center r =>
      simpa [address,result,expected,Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc]
        using hr.center (r,PUnit.unit)
    | denominator r =>
      simpa [address,result,expected,Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc]
        using hr.denominator hi (r,PUnit.unit)
    | output r d =>
      exact hr.output (r,d,PUnit.unit)
  · intro rg off hn
    symm
    apply hr.frame rg off
    · by_cases hm : rg="M"
      · right
        intro i heq
        apply hn
        refine ⟨.center i.1,trivial,?_⟩
        apply Prod.ext hm.symm
        simpa [Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc] using heq.symm
      · exact Or.inl hm
    · by_cases hf : c.finish=0
      · right
        by_cases hl : rg="L"
        · right
          intro i heq
          apply hn
          refine ⟨.denominator i.1,hf,?_⟩
          apply Prod.ext hl.symm
          simpa [Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc] using heq.symm
        · exact Or.inl hl
      · exact Or.inl hf
    · by_cases ho : rg="O"
      · right
        intro i heq
        apply hn
        refine ⟨.output i.1 i.2.1,trivial,?_⟩
        exact Prod.ext ho.symm heq.symm
      · exact Or.inl ho

theorem complete (c : Call) (h : Valid c) (a : Allocated c extent) (s : BlockState) :
    (∀ idx : GridIndex c.grid,c.source.TraceSafe extent (s.withGridIndex idx)) ∧
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary c.source c.grid s final) ∧
      (∀ idx i,active c i → final.readMem (address c idx i).1 (address c idx i).2=result c s idx i) ∧
      BlockState.WriteWithin (writes c) s final := by
  refine ⟨safe c h a s,?_⟩
  exact HostComposition.assemble c.grid c.source s (fun _ => active c) (address c) (result c s)
    (program c h s) (fun i j x y _ _ he=>(unique c h i j x y he).1)

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Whole

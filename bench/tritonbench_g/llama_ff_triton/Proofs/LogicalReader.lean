import bench.tritonbench_g.llama_ff_triton.Proofs.IOResults
import bench.tritonbench_g.llama_ff_triton.IO
namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.ReaderProof
open VeriTile Triton FP.Structural Execution ReaderIO
open scoped BigOperators LogicalDomainIO
noncomputable section

theorem target_eq (p : Parameters) : (view p).lift (fun xs row column =>
      let rho := 1 / Real.sqrt ((∑ k ∈ Finset.range p.inner, xs.arrays.x row.val k ^ 2) /
        (p.inner : ℝ) + p.epsilon)
      let u := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w1 column.val k) * rho
      let v := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w3 column.val k) * rho
      (u * (1 / (1 + Real.exp (-u)))) * v) = Reference.realFormula p := by
  have sum_eq (f : Nat → ℝ) (n : Nat) :
      (∑ k ∈ Finset.range n, f k) = Int8DequantMatmul.rsum f n := by
    induction n with
    | zero => simp [Int8DequantMatmul.rsum]
    | succ n ih => simp [Finset.sum_range_succ,Int8DequantMatmul.rsum,ih]
  funext xs o j
  fin_cases o
  change (let ys := ReaderIO.inputs p xs
    let rho := 1 / Real.sqrt ((∑ k ∈ Finset.range p.inner, ys.arrays.x j.1.val k ^ 2) /
      (p.inner : ℝ) + p.epsilon)
    let u := (∑ k ∈ Finset.range p.inner,
      ys.arrays.x j.1.val k * ys.arrays.rms k * ys.arrays.w1 j.2.1.val k) * rho
    let v := (∑ k ∈ Finset.range p.inner,
      ys.arrays.x j.1.val k * ys.arrays.rms k * ys.arrays.w3 j.2.1.val k) * rho
    (u * (1 / (1 + Real.exp (-u)))) * v) = _
  simp only [Reference.realFormula,LlamaFF.realTarget,Reference.realInput,
    ReaderIO.inputs,ReaderIO.decode,ReaderIO.backing,Equiv.coe_fn_mk,sum_eq]


/-- Exact full-domain judgment equivalence, retaining every source array and
all original safety, complete-grid execution and typed exterior frame clauses. -/
theorem judgment_iff (p : Parameters) :
    (view p ⊨{byteDomain} fun xs row column =>
      let rho := 1 / Real.sqrt ((∑ k ∈ Finset.range p.inner, xs.arrays.x row.val k ^ 2) /
        (p.inner : ℝ) + p.epsilon)
      let u := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w1 column.val k) * rho
      let v := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w3 column.val k) * rho
      (u * (1 / (1 + Real.exp (-u)))) * v) ↔
      (Interface.programIO Bool.false realArithmetic RealGrid.unusedGate p).ImplementsOn
        Interface.byteDomain (Reference.realFormula p) := by
  rw [LogicalDomainIO.implementsOn_iff,target_eq]
  rfl

theorem correct (p : Parameters) : (view p ⊨{byteDomain} fun xs row column =>
      let rho := 1 / Real.sqrt ((∑ k ∈ Finset.range p.inner, xs.arrays.x row.val k ^ 2) /
        (p.inner : ℝ) + p.epsilon)
      let u := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w1 column.val k) * rho
      let v := (∑ k ∈ Finset.range p.inner,
        xs.arrays.x row.val k * xs.arrays.rms k * xs.arrays.w3 column.val k) * rho
      (u * (1 / (1 + Real.exp (-u)))) * v) :=
  (judgment_iff p).mpr (Interface.Result.real p)
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.ReaderProof

import bench.repairs.tritonbench_g.kldiv_ops.Proofs.BackwardValues
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.ByteRealization

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section

/-- The complete backward source makes physical stores only to the target
prefix. This closes the byte-level realization of an oversized work tensor. -/
theorem BackwardResult.source_bytes {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
    {M : Algebra Word} {api : Services Word} {compiler : Compiler}
    {target grad dst : Tensor} {flag : Bool} {entry : State} {returned : Tensor} {final : State}
    (f : BackwardResult codec num M api compiler target grad dst flag entry returned final)
    (lossless : Lossless codec) :
    f.trace.launched.heap=writeValues codec f.launch.plan.arena.output (List.range target.numel)
      (fun i=>(f.launch.final.mem "O" i).read .real) f.trace.ready.heap := by
  rw [f.launch.heap]
  apply commit_prefix codec lossless
  · change target.numel ≤ f.trace.work.numel
    simpa [Tensor.numel,f.workShape] using f.capacity
  · intro i lo hi
    rw [f.launch.model]
    have tail := Backward.grid_tail M (params f.trace.rows f.trace.cols) flag
      (represented codec f.launch.plan.arena f.trace.ready.heap).mem i (by
        simpa [params,Tensor.numel,f.targetShape] using lo)
    dsimp only [params] at tail ⊢
    rw [tail]
    rfl

/-- Forward writes every logical output word and no allocation padding. -/
theorem ForwardResult.source_bytes {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
    {M : Algebra Word} {api : Services Word} {compiler : Compiler}
    {prediction target : Tensor} {flag : Bool} {name : String} {eps : Word}
    {entry : State} {returned : Tensor} {final : State}
    (f : ForwardResult codec num M api compiler prediction target flag name eps entry returned final) :
    f.trace.launched.heap=writeValues codec f.launch.plan.arena.output (List.range f.trace.out.numel)
      (fun i=>(f.launch.final.mem "O" i).read .real) f.trace.ready.heap := by
  exact f.launch.heap

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host

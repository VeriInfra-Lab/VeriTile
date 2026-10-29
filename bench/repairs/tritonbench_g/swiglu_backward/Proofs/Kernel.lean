import bench.repairs.tritonbench_g.swiglu_backward.Kernels
import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Index

namespace VeriTile.Bench.Repairs.SwigluBackward.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section

/-- The selected entry is exactly the readable canonical typed Triton
quotation, including all three FP32 casts and both constexpr blocks. -/
theorem selected_quotation (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy cols block : Nat) (recompute : Bool) :
    kernel X Y DOUT OUT DX DY sx sy sd so sdx sdy cols block recompute =
      swigluBackward%[tl.float32,tl.float32,tl.float32,tl.float32,original]
        X Y DOUT OUT DX DY sx sy sd so sdx sdy cols block recompute := by rfl

/-- Only the supplied kernel grid executes here. Preparation, autotune trials,
and returned wrapper objects are outside this entry. -/
def run [Inhabited α] (M : Algebra α) (p : Prepared α call entry preexisting)
    (config : Config) : List (Nat × Nat) → State α → Option (State α)
  | [], s => some s
  | (row,tile)::rest, s =>
      (exec M (kernel p.layout.X p.layout.Y p.layout.DOUT p.layout.OUT
        p.layout.DX p.layout.DY p.layout.sx p.layout.sy p.layout.sd
        p.layout.so p.layout.sdx p.layout.sdy p.ncols config.block call.recompute)
        (tileState M p.layout config.block row tile s)).bind (run M p config rest)

theorem run_eq [Inhabited α] (M : Algebra α) (p : Prepared α call entry preexisting)
    (config : Config) (tiles : List (Nat × Nat)) (s : State α) :
    run M p config tiles s = runTiles M p.layout config.block call.recompute tiles s := by
  induction tiles generalizing s with
  | nil => rfl
  | cons tile rest ih =>
      cases tile
      simp only [run,runTiles,tileExec,FP.Structural.exec,Layout.source,kernel,sourceKernel,ih]
      rfl

/-- Full effective output and frame, stated using logical reads of the
original tensors. The paired chunks share their actual backing allocation. -/
def Result [Inhabited α] (f : Channel → α → α → α → α)
    (p : Prepared α call entry preexisting) (final : State α) : Prop :=
  (∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols →
    p.commit final (p.location ch r q).1 (p.location ch r q).2 = f ch
      (call.input.read entry (r*(2*p.ncols)+q))
      (call.input.read entry (r*(2*p.ncols)+(p.ncols+q)))
      (call.upstream.read entry (r*p.ncols+q))) ∧
  (∀ region a, WholeOutside p.layout call.recompute region a →
    final.mem region a = p.state.mem region a) ∧
  (∀ allocation a, PhysicalOutside p allocation a →
    p.commit final allocation a = p.memory allocation a) ∧
  (∀ r q, r < p.batchShape.prod → q < 2*p.ncols →
    p.xy.matrix.read (p.commit final) r q = p.xy.matrix.read p.memory r q) ∧
  (∀ r q, r < p.batchShape.prod → q < p.ncols →
    p.dout.matrix.read (p.commit final) r q = p.dout.matrix.read p.memory r q)

theorem success [Inhabited α] (M : Algebra α)
    (p : Prepared α call entry preexisting) (storage : Storage p)
    (shapes : checks p = Bool.true) (admitted : checkLaunch p storage.byteBase = Bool.true)
    (config : Config) (tiles : List (Nat × Nat))
    (schedule : tiles.Perm (grid p.layout config.block)) :
    ∃ final, run M p config tiles p.state = some final ∧
      Result (scalar M) p final ∧ CheckedIndices p storage := by
  let domain := checkedDomain p storage shapes ((checkLaunch_iff p storage.byteBase).mp admitted)
  obtain ⟨final,ran,values,frame⟩ := grid_result M p.layout config.block call.recompute
    config.positive (Domain.ownership p domain) tiles schedule p.state
  refine ⟨final,?_,?_,checked_indices p storage shapes ((checkLaunch_iff _ _).mp admitted)⟩
  · rw [run_eq]; exact ran
  · refine ⟨?_,frame,commit_frame p final,live_inputs_preserved p domain final⟩
    intro ch on r q hr hq
    rw [commit_value p domain final ch on r q hr hq]
    have value := values ch on r q hr hq
    have reads := prepared_reads p domain r q hr hq
    rw [reads.1,reads.2.1,reads.2.2] at value
    rw [value]
    rfl

#axiomsClean success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.KernelProof

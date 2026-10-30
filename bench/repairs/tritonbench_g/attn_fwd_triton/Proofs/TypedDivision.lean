import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedHost

/-! The public candidate executes division. This composition reuses complete
source execution and frames without any reciprocal rewrite model or domain.
The reference is the same division invocation on the original initial inputs. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedPhysical (Physical pull)
open Verified.PhysicalCells (Layout Role FreshBytes fresh_cells)
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

/-- The actual division source grid succeeds, reads back its initial-input
division reference, and preserves the full complement of its output window. -/
theorem division_grid {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (initial : State α)
    (hD : p.headDim≤BD) (hn : NonOverlapping p) :
    ∃ final,
      Nonempty (GridRun A p BD Bool.false initial final) ∧
      (∀ c : Coordinate p, final.mem p.Out (logicalAddress p c)=reference A p BD support initial c) ∧
      (∀ region offset, ¬Window p region offset → final.mem region offset=initial.mem region offset) := by
  have hprograms : ∀ idx : ProgramIndex p, ∃ t,
      exec (engine A) (source p BD Bool.false) (atProgram p initial idx)=some t ∧
      Frame p BD idx.1.val idx.2.val (atProgram p initial idx) t := by
    intro idx
    exact executes_framed A p BD support Bool.false (atProgram p initial idx)
  choose programs hexec hframe using hprograms
  let final : State α := {initial with mem := fun region offset =>
    if h : (∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset) then
      ((programs h.choose).mem region offset) else (initial.mem region offset)}
  have hread : ∀ (idx : ProgramIndex p) region offset, Written p BD idx.1.val idx.2.val region offset →
      final.mem region offset=(programs idx).mem region offset := by
    intro idx region offset hw
    have hex : ∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset := ⟨idx,hw⟩
    change (if h : (∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset) then
      ((programs h.choose).mem region offset) else (initial.mem region offset))=_
    rw [dif_pos hex,disjoint p BD hn hex.choose idx region offset hex.choose_spec hw]
  have hf : ∀ region offset, ¬Window p region offset → final.mem region offset=initial.mem region offset := by
    intro region offset hnot
    change (if h : (∃ idx : ProgramIndex p, Written p BD idx.1.val idx.2.val region offset) then
      ((programs h.choose).mem region offset) else (initial.mem region offset))=_
    apply dif_neg
    rintro ⟨idx,hw⟩
    exact hnot (written_within p BD idx region offset hw)
  refine ⟨final,⟨⟨programs,hexec,hframe,disjoint p BD hn,hread,hf⟩⟩,?_,hf⟩
  intro c
  rw [hread (owner p c) _ _ (covered p BD hD c)]
  have he := Option.some.inj ((hexec (owner p c)).symm.trans
    (reference_exec A p BD support initial (owner p c)))
  rw [he]
  rfl

/-- Physical readback and full memory frame for the actual division grid. -/
theorem physical_division_grid {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD)
    (a : Layout) (base : Role → Nat) (views : Views a p base)
    (initial : Physical α a.toLayout) (template : State α)
    (hD : p.headDim≤BD) (hn : NonOverlapping p)
    (fresh : FreshBytes a (fun k => Window p k.1 k.2)) :
    ∃ final, ∃ physical : Physical α a.toLayout,
      Nonempty (GridRun A p BD Bool.false (pull a.toLayout initial template) final) ∧
      pull a.toLayout physical final=final ∧
      (∀ c : Coordinate p, physical (outputCell a p base views c)=
        reference A p BD support (pull a.toLayout initial template) c) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, Window p k.1 k.2 → a.address k≠cell.val) → physical cell=initial cell) := by
  obtain ⟨final,ho,hread,hframe⟩ := division_grid A p BD support
    (pull a.toLayout initial template) hD hn
  let physical := TypedRealisation.realise a.toLayout initial final (fun k => Window p k.1 k.2)
  have hpull : pull a.toLayout physical final=final :=
    TypedRealisation.realise_pull_state a.toLayout initial template final _ (fresh_cells a _ fresh)
      (fun k hk => hframe k.1 k.2 hk)
  refine ⟨final,physical,ho,hpull,?_,?_⟩
  · intro c
    rw [←output_read a p base views physical c]
    change (pull a.toLayout physical final).mem p.Out (logicalAddress p c)=_
    rw [hpull]
    exact hread c
  · exact fun cell h => TypedRealisation.realise_unwritten a.toLayout initial final _ cell h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedHost
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd.TypedPhysical (Physical pull)
open Verified.PhysicalCells (Layout)
open Verified.HostSetup (Call calls beforeEmpty nonemptyChecks)
open TypedExecution HostShape
noncomputable section

/-- Ordinary host return for the frozen division candidate. No arithmetic
identity, finite/nonzero restriction, or reciprocal execution is a premise. -/
theorem division_successful_return {α : Type} [Inhabited α] (A : Algebra α)
    (request : Request α) (a : Layout) (caller : Physical α a.toLayout)
    (ret : Prepared request a caller) (template : State α) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
    (hlaunch : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim))
    (support : DotSupport A (blockD ret.p.headDim)) :
    setup ret ok=.launch ret.returnedOutput rfl hne hlaunch ∧ NativeCertificate ret ∧
    (ret.p.context≤274877906816 ∧ ret.p.batches*ret.p.heads≤65535 ∧ blockD ret.p.headDim≤1048576) ∧
    ∃ final, ∃ physical : Physical α a.toLayout,
      Nonempty (GridRun A ret.p (blockD ret.p.headDim) Bool.false (pull a.toLayout ret.initial template) final) ∧
      pull a.toLayout physical final=final ∧
      (∀ c : Coordinate ret.p, physical (outputCell a ret.p ret.base ret.views c)=
        reference A ret.p (blockD ret.p.headDim) support (pull a.toLayout ret.initial template) c) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, Window ret.p k.1 k.2 → a.address k≠cell.val) → physical cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.original.owned cell → physical cell=caller cell) := by
  refine ⟨by simp [setup,hbefore,hscales,hcompile,hne,hlaunch],native_certificate ret,
    ⟨NativeLaunch.context _ _ hlaunch,hlaunch.gridY,hlaunch.arange⟩,?_⟩
  obtain ⟨final,physical,ho,hpull,hread,hframe⟩ := physical_division_grid A ret.p
    (blockD ret.p.headDim) support a ret.base ret.views ret.initial template (blockD_covers _)
    ret.nonoverlap ret.fresh
  refine ⟨final,physical,ho,hpull,hread,fun cell hp hw =>
    (hframe cell hw).trans (ret.preparationFrame cell hp),?_⟩
  intro cell hc
  exact (hframe cell (ret.outputFresh cell hc)).trans
    (ret.preparationFrame cell (ret.preparationFresh cell hc))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedHost

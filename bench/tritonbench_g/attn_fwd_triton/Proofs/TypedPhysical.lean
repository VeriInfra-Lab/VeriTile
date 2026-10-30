import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedGrid
import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedRealisation

namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedPhysical (Physical pull)
open Verified.PhysicalCells (Layout Role width FreshBytes fresh_cells)
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

def roleRegion (p : Config) : Role → RegionName
  | .query => p.Q | .key => p.K | .value => p.V | .output => p.Out
  | .queryScale => p.QS | .keyScale => p.KS

def roleOffset (p : Config) (z h n d : Nat) : Role → Nat
  | .query => z*p.queryBatch+h*p.queryHead+n*p.queryRow+d*p.queryDim
  | .key => z*p.keyBatch+h*p.keyHead+n*p.keyRow+d*p.keyDim
  | .value => z*p.valueBatch+h*p.valueHead+n*p.valueRow+d*p.valueDim
  | .output => z*p.outputBatch+h*p.outputHead+n*p.outputRow+d*p.outputDim
  | .queryScale | .keyScale => n

def roleActive (p : Config) (z h n d : Nat) : Role → Prop
  | .queryScale => z=0 ∧ h=0 ∧ d=0 ∧ n<p.batches*p.heads*((p.context+127)/128)
  | .keyScale => z=0 ∧ h=0 ∧ d=0 ∧ n<p.batches*p.heads*((p.context+63)/64)
  | _ => z<p.batches ∧ h<p.heads ∧ n<p.context ∧ d<p.headDim

def roleKey (p : Config) (z h n d : Nat) (r : Role) : AliasMemory.Key :=
  (roleRegion p r,roleOffset p z h n d r)

/-- One physical initial memory realizes all six actual source roles. Read
views may overlap; full byte intervals protect fresh output stores. -/
structure Views (a : Layout) (p : Config) (base : Role → Nat) : Prop where
  live : ∀ z h n d r, roleActive p z h n d r → a.live (roleKey p z h n d r)
  address : ∀ z h n d r, roleActive p z h n d r →
    a.address (roleKey p z h n d r)=base r+width r*roleOffset p z h n d r
  widths : ∀ z h n d r, roleActive p z h n d r → a.width (roleKey p z h n d r)=width r

def outputCell (a : Layout) (p : Config) (base : Role → Nat) (v : Views a p base)
    (c : Coordinate p) : Fin a.capacity :=
  ⟨base .output+width .output*logicalAddress p c,by
    rw [show logicalAddress p c=roleOffset p c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val .output from rfl,
      ←v.address _ _ _ _ .output ⟨c.1.isLt,c.2.1.isLt,c.2.2.1.isLt,c.2.2.2.isLt⟩]
    exact a.bounded _ (v.live _ _ _ _ .output ⟨c.1.isLt,c.2.1.isLt,c.2.2.1.isLt,c.2.2.2.isLt⟩)⟩

theorem output_read (a : Layout) (p : Config) (base : Role → Nat) (v : Views a p base)
    (memory : Physical α a.toLayout) (c : Coordinate p) :
    TypedPhysical.read a.toLayout memory (p.Out,logicalAddress p c)=memory (outputCell a p base v c) := by
  have hl := v.live c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val .output
    ⟨c.1.isLt,c.2.1.isLt,c.2.2.1.isLt,c.2.2.2.isLt⟩
  change a.live (p.Out,logicalAddress p c) at hl
  simp only [TypedPhysical.read,dif_pos hl]
  congr 1
  apply Fin.ext
  exact v.address _ _ _ _ .output ⟨c.1.isLt,c.2.1.isLt,c.2.2.1.isLt,c.2.2.2.isLt⟩

/-- Caller memory, actual complete source grids, physical BF16-cell readback
against the unchanged division reference on initial inputs, and full frame.
No execution, returned-value, or program-bridge premise is supplied. -/
theorem physical_grid_equivalence {α : Type} [Inhabited α] (R : FPEquivProofs.Rules)
    (A : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions A D)
    (p : Config) (BD : Nat) (support : DotSupport A BD)
    (a : Layout) (base : Role → Nat) (views : Views a p base)
    (initial : Physical α a.toLayout) (template : State α)
    (hD : p.headDim≤BD) (hn : NonOverlapping p)
    (fresh : FreshBytes a (fun k => Window p k.1 k.2))
    (hd : ∀ idx : ProgramIndex p, ∀ site∈(program p BD Bool.false).domain,
      site.Holds A D (atProgram p (pull a.toLayout initial template) idx)) :
    ∃ final, ∃ physical : Physical α a.toLayout,
      Nonempty (GridRun A p BD Bool.false (pull a.toLayout initial template) final) ∧
      Nonempty (GridRun A p BD Bool.true (pull a.toLayout initial template) final) ∧
      pull a.toLayout physical final=final ∧
      (∀ c : Coordinate p, physical (outputCell a p base views c)=
        reference A p BD support (pull a.toLayout initial template) c) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, Window p k.1 k.2 → a.address k≠cell.val) → physical cell=initial cell) := by
  obtain ⟨final,ho,hr,hread,hframe⟩ := grid_equivalence R A D hM p BD support
    (pull a.toLayout initial template) hD hn hd
  let physical := TypedRealisation.realise a.toLayout initial final (fun k => Window p k.1 k.2)
  have hpull : pull a.toLayout physical final=final :=
    TypedRealisation.realise_pull_state a.toLayout initial template final _ (fresh_cells a _ fresh)
      (fun k hk => hframe k.1 k.2 hk)
  refine ⟨final,physical,ho,hr,hpull,?_,?_⟩
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

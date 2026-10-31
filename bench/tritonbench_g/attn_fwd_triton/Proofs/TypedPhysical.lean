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

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

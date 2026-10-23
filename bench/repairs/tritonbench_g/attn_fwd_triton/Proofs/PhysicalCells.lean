import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.GridLaunch
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.ByteLayout

/-! Byte-addressed typed-cell realization. Read views may alias; fresh output
byte intervals are separate. This is not a raw-byte codec or GPU model. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.PhysicalCells
open VeriTile.Triton Memory Relocation
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
open scoped Classical

def region (p : Parameters) : Role → RegionName
  | .query => p.Q | .key => p.K | .value => p.V | .output => p.Out
  | .queryScale => p.QS | .keyScale => p.KS

def offset (p : Parameters) (z h n d : Nat) : Role → Nat
  | .query => RealProofs.address p.queryBatch p.queryHead p.queryRow p.queryDim z h n d
  | .key => RealProofs.address p.keyBatch p.keyHead p.keyRow p.keyDim z h n d
  | .value => RealProofs.address p.valueBatch p.valueHead p.valueRow p.valueDim z h n d
  | .output => RealProofs.address p.outputBatch p.outputHead p.outputRow p.outputDim z h n d
  | .queryScale | .keyScale => n

def active (p : Parameters) (z h n d : Nat) : Role → Prop
  | .queryScale => z=0 ∧ h=0 ∧ d=0 ∧ n<p.batches*p.heads*((p.context+127)/128)
  | .keyScale => z=0 ∧ h=0 ∧ d=0 ∧ n<p.batches*p.heads*((p.context+63)/64)
  | _ => z<p.batches ∧ h<p.heads ∧ n<p.context ∧ d<p.headDim

def key (A : FlatAlloc) (p : Parameters) (z h n d : Nat) (r : Role) : AliasMemory.Key :=
  (A.flat,offset p z h n d r+A.base (region p r))

/-- Tensor-library view metadata binds all six source roles to one initial
physical state, retaining storage-offset bases and the actual element widths. -/
structure Views (a : Layout) (A : FlatAlloc) (p : Parameters) (base : Role → Nat) : Prop where
  live : ∀ z h n d r, active p z h n d r → a.live (key A p z h n d r)
  address : ∀ z h n d r, active p z h n d r →
    a.address (key A p z h n d r)=base r+width r*offset p z h n d r
  widths : ∀ z h n d r, active p z h n d r → a.width (key A p z h n d r)=width r

theorem active_read (a : Layout) (A : FlatAlloc) (p : Parameters) (base : Role → Nat)
    (v : Views a A p base) (initial : AliasMemory.Physical a.toLayout)
    (z h n d : Nat) (r : Role) (hr : active p z h n d r) :
    AliasMemory.read a.toLayout initial (key A p z h n d r) =
      initial ⟨base r+width r*offset p z h n d r,by
        rw [←v.address z h n d r hr]
        exact a.bounded _ (v.live z h n d r hr)⟩ := by
  simp only [AliasMemory.read,dif_pos (v.live z h n d r hr)]
  congr 1
  apply Fin.ext
  exact v.address z h n d r hr

/-- Actual whole-grid source execution, physical readback, and frame outside
all output byte-cell starts, with no execution or output premise. -/
theorem launch (a : Layout) (A : FlatAlloc) (p : Parameters) (BD : Nat)
    (initial : AliasMemory.Physical a.toLayout) (template : BlockState)
    (hD : p.headDim ≤ BD) (hstage : p.stage=3) (reciprocal : Bool)
    (hn : OutputLayout.NonOverlapping p) (hf : FreshBytes a (GridFootprint.window A p)) :
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p BD reciprocal) (GridCoordinates.grid p)
        (AliasMemory.pull a.toLayout initial template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        (AliasMemory.pull a.toLayout final t).readMem A.flat (OutputLayout.address p i+A.base p.Out)=
          GridLaunch.expected A (AliasMemory.pull a.toLayout initial template) p i) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=initial cell) := by
  let s := AliasMemory.pull a.toLayout initial template
  obtain ⟨t,ht,hr,hframe⟩ := GridLaunch.launch A s p BD hD hstage reciprocal hn
  let final := AliasMemory.realise a.toLayout initial t (GridFootprint.window A p)
  have hpull : AliasMemory.pull a.toLayout final t=t :=
    AliasMemory.realise_pull_state a.toLayout initial template t (GridFootprint.window A p)
      (fresh_cells a _ hf) (fun k hk => hframe k.1 k.2 hk)
  refine ⟨t,final,ht,?_,?_⟩
  · intro i
    rw [hpull]
    exact hr i
  · exact fun cell h => AliasMemory.realise_unwritten a.toLayout initial t (GridFootprint.window A p) cell h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.PhysicalCells

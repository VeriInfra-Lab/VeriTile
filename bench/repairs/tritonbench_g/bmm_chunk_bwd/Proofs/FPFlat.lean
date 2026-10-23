import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPLogical

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPFlat
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
variable {α : Type} [Inhabited α]
open Classical

/-- Runtime pointer bindings. Different read-only source roles may designate
the same physical region/window. Symbolic role names do not assert allocation
disjointness. Offsets are FP32 element offsets; NativeAddress scales by four. -/
structure Placement where
  region : RegionName → RegionName
  base : RegionName → Nat
  extent : RegionName → Nat

def address (a : Placement) (r : RegionName) (off : Nat) := a.base r+off

def scratchCell (a : Placement) (out : RegionName) (r : RegionName) (off : Nat) : Prop :=
  r = a.region out ∧ a.base out ≤ off ∧ off < a.base out+a.extent out

def lift (h : HostCall) (a : Placement) (s : State α) : State α :=
  {s with mem := fun r off =>
    if r = h.A ∨ r = h.Dout then .mk .real ((s.mem (a.region r) (address a r off)).read .fp32)
    else .mk .fp32 ((s.mem (a.region r) (address a r off)).read .fp32)}

/-- Reify the named output allocation into its finite physical window.
Only source Out is mutable. Source frame and store commutation below justify
this memory representation; it is not an assumed kernel-output equation. -/
def publish (a : Placement) (out : RegionName) (physical source : State α) : State α :=
  {physical with mem := fun r off => if scratchCell a out r off then source.mem out (off-a.base out) else physical.mem r off}

theorem lift_input (h : HostCall) (a : Placement) (s : State α) (r : RegionName) (off : Nat)
    (hr : r = h.A ∨ r = h.Dout) :
    ((lift h a s).mem r off).read .real = (s.mem (a.region r) (address a r off)).read .fp32 := by
  simp [lift,hr]

theorem lift_fp32 (h : HostCall) (a : Placement) (s : State α) (r : RegionName) (off : Nat)
    (hr : r ≠ h.A ∧ r ≠ h.Dout) :
    ((lift h a s).mem r off).read .fp32 = (s.mem (a.region r) (address a r off)).read .fp32 := by
  simp [lift,hr.1,hr.2]

theorem publish_read (a : Placement) (out : RegionName) (physical source : State α) (off : Nat)
    (bound : off < a.extent out) :
    (publish a out physical source).mem (a.region out) (address a out off) = source.mem out off := by
  simp [publish,scratchCell,address,bound]

theorem publish_frame (a : Placement) (out : RegionName) (physical source : State α)
    (r : RegionName) (off : Nat) (outside : ¬ scratchCell a out r off) :
    (publish a out physical source).mem r off = physical.mem r off := by simp [publish,outside]

/-- Every actual typed output store is the same physical FP32 cell write.
Inactive lanes perform neither write in this representation. -/
theorem write_commutes (a : Placement) (out : RegionName) (physical source : State α)
    (off : Nat) (v : Cell α) (bound : off < a.extent out) :
    (publish a out physical (source.write out off v)).mem =
      ((publish a out physical source).write (a.region out) (address a out off) v).mem := by
  funext r n
  by_cases hh : r = a.region out ∧ n = address a out off
  · rcases hh with ⟨rfl,rfl⟩
    simp [publish_read a out physical _ off bound,State.write_same]
  · by_cases hs : scratchCell a out r n
    · have hne : n-a.base out ≠ off := by
        intro he
        apply hh
        refine ⟨hs.1,?_⟩
        unfold address
        have hb := hs.2.1
        omega
      simp [publish,hs,State.write,hh,hne]
    · have hne : ¬ (r = a.region out ∧ n = address a out off) := hh
      simp [publish,hs,State.write,hne]

def outOffset (h : HostCall) (c : Tuning) (q : HostFP.Pid) (s : State α) (i : TileIndex [c.BM,c.BN]) :=
  FPReference.offset (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h)
    h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature i

theorem output_bound (h : HostCall) (c : Tuning) (alloc : FlatAlloc)
    (layout : HostContract.Layout h c alloc) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c)
    (s : State α) (i : TileIndex [c.BM,c.BN])
    (hi : FPReference.active (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h) i) :
    outOffset h c q s i < alloc.extent h.Out := by
  exact (layout.bounds _ (FPGridOutput.grid_member h c q hq) default).output i hi

#axiomsClean write_commutes
#axiomsClean output_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPFlat

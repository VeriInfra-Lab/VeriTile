import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPFlat

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPRoles
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPLogical
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 3000000

def tensorOffset (h : HostCall) (st : TypedSource.TensorStrides) (x : Coordinate) :=
  x.batch*st.batch+x.sequence*st.sequence+x.group*(h.headStride st.head)+x.feature*st.feature

def doutOffset (h : HostCall) (b c g k r : Nat) :=
  b*h.doutStrides.batch+c*h.doutStrides.chunk+g*(h.headStride h.doutStrides.head)+
    k*h.doutStrides.row+r*h.doutStrides.column

/-- Generic logical-to-physical input binding. The direct boundary supplies
original admitted FP32 views, preserving every input coordinate, not
just a fixture, output, reduction result, or successful kernel execution.
Read-only role windows may overlap each other and the caller output. -/
structure WholeInputs (neg : α → α) (h : HostCall) (caller : Caller)
    (before physical : State α) (a : FPFlat.Placement) : Prop where
  tensor : ∀ x, valid h x →
    (physical.mem (a.region h.A) (FPFlat.address a h.A (tensorOffset h h.aStrides x))).read .fp32 =
      tensorRead neg caller.a before x
  dout : ∀ b c g k r, b < h.batch → c < h.nchunks → g < h.groupCount → k < h.chunk → r < h.chunk →
    (physical.mem (a.region h.Dout) (FPFlat.address a h.Dout (doutOffset h b c g k r))).read .fp32 =
      doutRead neg caller.dout before b c g k r
  residual : ∀ x, valid h x → h.hasResidual = Bool.true →
    (physical.mem (a.region h.Residual) (FPFlat.address a h.Residual (tensorOffset h h.residualStrides x))).read .fp32 =
      tensorRead neg caller.residual before x

/-- Re-tagging uses distinct symbolic roles for residual storage and matrix
operand carriers. This is not physical allocation separation. -/
def ResidualTag (h : HostCall) := h.Residual ≠ h.A ∧ h.Residual ≠ h.Dout

theorem grid_coordinates (h : HostCall) (c : Tuning) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c) :
    q.2.1 < h.batch ∧ chunk h q < h.nchunks ∧ head h q < h.groupCount := by
  have hh := (HostGrid.mem_programs _ _).mp (FPGridOutput.grid_member h c q hq)
  have hg : q.2.2 < h.nchunks*h.groupCount := hh.2.2
  have gp : 0 < h.groupCount := by
    by_contra hn
    have hz : h.groupCount = 0 := by omega
    simp [hz] at hg
  refine ⟨hh.2.1,(Nat.div_lt_iff_lt_mul gp).mpr hg,?_⟩
  have he : head h q = q.2.2 % h.groupCount := by
    unfold head chunk
    have hh := Nat.mod_add_div q.2.2 h.groupCount
    rw [Nat.mul_comm h.groupCount] at hh
    omega
  rw [he]
  exact Nat.mod_lt _ gp

theorem contraction_coordinate (h : HostCall) (c : Tuning) (q : HostFP.Pid)
    (hq : q ∈ HostFP.grid h c) (step lane n : Nat)
    (hk : lane < length h q-step*c.BCS) (hn : n < h.K) :
    valid h ⟨q.2.1,chunk h q*h.chunk+step*c.BCS+lane,head h q,n⟩ := by
  have hh := grid_coordinates h c q hq
  have hs := Nat.min_le_right h.chunk (h.seqlen-chunk h q*h.chunk)
  change _ ≤ _ at hs
  unfold length at hk
  refine ⟨hh.1,?_,hh.2.2,hn⟩
  change chunk h q*h.chunk+step*c.BCS+lane < h.seqlen
  omega

theorem output_coordinate (h : HostCall) (c : Tuning) (q : HostFP.Pid)
    (hq : q ∈ HostFP.grid h c) (s : State α) (i : TileIndex [c.BM,c.BN])
    (hi : FPReference.active (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h) i) :
    valid h (coord h c q i) := by
  have hh := grid_coordinates h c q hq
  have hs := Nat.min_le_right h.chunk (h.seqlen-chunk h q*h.chunk)
  change row h c q i.1 < length h q ∧ column h c q i.2.1 < h.K at hi
  change q.2.1 < h.batch ∧ chunk h q*h.chunk+row h c q i.1 < h.seqlen ∧
    head h q < h.groupCount ∧ column h c q i.2.1 < h.K
  exact ⟨hh.1,by unfold length at hi; omega,hh.2.2,hi.2⟩

/-- Every actual source load is derived from the corresponding whole-view
contract, including padded Dout rows loaded by the source's row<chunk mask. -/
theorem pins (neg : α → α) (h : HostCall) (c : Tuning) (caller : Caller)
    (before physical : State α) (a : FPFlat.Placement) (tag : ResidualTag h)
    (whole : WholeInputs neg h caller before physical a) :
    InputPins neg h c caller before (FPFlat.lift h a physical) := by
  constructor
  · intro q hq step i hr hk
    rw [FPFlat.lift_input h a physical h.Dout _ (Or.inr rfl)]
    have hh := grid_coordinates h c q hq
    have kc : step*c.BCS+i.2.1.val < h.chunk := by
      have hl := Nat.min_le_left h.chunk (h.seqlen-chunk h q*h.chunk)
      change length h q ≤ h.chunk at hl
      omega
    convert whole.dout q.2.1 (chunk h q) (head h q) (step*c.BCS+i.2.1.val)
      (row h c q i.1) hh.1 hh.2.1 hh.2.2 kc hr using 1 <;>
      simp [doutOffset,FPReference.dBase,FPReference.chunk,FPReference.head,
        HostFP.launch,HostContract.parameters,chunk,head,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  · intro q hq step i hk hn
    rw [FPFlat.lift_input h a physical h.A _ (Or.inl rfl)]
    have hx := contraction_coordinate h c q hq step i.1.val (column h c q i.2.1) hk hn
    convert whole.tensor _ hx using 1 <;>
      simp [tensorOffset,FPReference.aBase,FPReference.chunk,FPReference.head,
        HostFP.launch,HostContract.parameters,chunk,head,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  · intro q hq i hi hres
    rw [FPFlat.lift_fp32 h a physical h.Residual _ tag]
    have hx := output_coordinate h c q hq (FPFlat.lift h a physical) i hi
    convert whole.residual _ hx hres using 1
    simp [tensorOffset,FPReference.offset,FPReference.chunk,FPReference.head,FPReference.row,
      FPReference.column,HostFP.launch,HostContract.parameters,coord,Boundary.coord,qReal,
      RealViews.row,RealViews.column,RealViews.chunk,RealViews.head,RealSetup.tilesN,TypedSource.HostCall.resStride,hres,
      Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

#axiomsClean pins
#axiomsClean output_coordinate
#specNonCircular WholeInputs avoiding [HostFP.run,FP.Structural.exec,FPReference.expected]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPRoles

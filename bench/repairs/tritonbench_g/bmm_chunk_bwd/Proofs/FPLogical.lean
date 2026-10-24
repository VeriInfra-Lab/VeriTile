import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPGridOutput
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Boundary

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPLogical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def logical (neg : α → α) (negative : Bool) (x : α) : α := if negative then neg x else x

def tensorRead (neg : α → α) (v : TensorView) (s : State α) (x : Coordinate) : α :=
  logical neg v.negative ((s.mem v.region (v.address x)).read .fp32)

def doutRead (neg : α → α) (v : DoutView) (s : State α) (b c g k r : Nat) : α :=
  logical neg v.negative ((s.mem v.region (v.address b c g k r)).read .fp32)

def qReal (q : HostFP.Pid) : HostGrid.Pid := ⟨q.1,q.2.1,q.2.2⟩
def row (h : HostCall) (c : Tuning) (q : HostFP.Pid) (i : Fin c.BM) :=
  RealViews.row c.BM c.BN (HostContract.parameters h) q.1 i
def column (h : HostCall) (c : Tuning) (q : HostFP.Pid) (j : Fin c.BN) :=
  RealViews.column c.BN (HostContract.parameters h) q.1 j
def chunk (h : HostCall) (q : HostFP.Pid) := q.2.2/h.groupCount
def head (h : HostCall) (q : HostFP.Pid) := q.2.2-chunk h q*h.groupCount
def length (h : HostCall) (q : HostFP.Pid) := min h.chunk (h.seqlen-chunk h q*h.chunk)
def coord (h : HostCall) (c : Tuning) (q : HostFP.Pid) (i : TileIndex [c.BM,c.BN]) :=
  Boundary.coord h c (qReal q) i

def leftBlock (M : Algebra α) (neg : α → α) (h : HostCall) (c : Tuning)
    (caller : Caller) (before : State α) (q : HostFP.Pid) (step : Nat) : Values α .real [c.BM,c.BCS] := fun i =>
  M.cast (some .fp32) .real .real
    (if row h c q i.1 < h.chunk ∧ i.2.1.val < length h q-step*c.BCS then
      doutRead neg caller.dout before q.2.1 (chunk h q) (head h q) (step*c.BCS+i.2.1.val) (row h c q i.1)
    else M.literal (some .fp32) .real 0)

def rightBlock (M : Algebra α) (neg : α → α) (h : HostCall) (c : Tuning)
    (caller : Caller) (before : State α) (q : HostFP.Pid) (step : Nat) : Values α .real [c.BCS,c.BN] := fun i =>
  M.cast (some .fp32) .real .real
    (if i.1.val < length h q-step*c.BCS ∧ column h c q i.2.1 < h.K then
      tensorRead neg caller.a before ⟨q.2.1,chunk h q*h.chunk+step*c.BCS+i.1.val,head h q,column h c q i.2.1⟩
    else M.literal (some .fp32) .real 0)

def carry (M : Algebra α) (D : DotPrimitive α) (neg : α → α) (h : HostCall) (c : Tuning)
    (caller : Caller) (before : State α) (q : HostFP.Pid) : Nat → Values α .real [c.BM,c.BN]
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | step+1 => fun i => M.binary (some .fp32) .real .add
      (carry M D neg h c caller before q step i)
      (D.plain (some .fp32) (batch := []) (m := c.BM) (k := c.BCS) (n := c.BN)
        (leftBlock M neg h c caller before q step) (rightBlock M neg h c caller before q step) i)

def expected (M : Algebra α) (D : DotPrimitive α) (neg : α → α) (h : HostCall) (c : Tuning)
    (caller : Caller) (before : State α) (q : HostFP.Pid) (i : TileIndex [c.BM,c.BN]) : α :=
  let acc := carry M D neg h c caller before q ((length h q+c.BCS-1)/c.BCS) i
  let acc := if h.hasResidual then M.binary (some .fp32) .real .add acc
    (M.cast (some .fp32) .fp32 .real (tensorRead neg caller.residual before (coord h c q i))) else acc
  M.cast (some .fp32) .real .fp32 acc

/-- Local coordinate-preserving preparation contracts, before any floating
cast or dot. Every actually loaded element of every role is pinned to the
initial logical caller view; padded lanes impose no input read obligation. -/
structure InputPins (neg : α → α) (h : HostCall) (c : Tuning) (caller : Caller)
    (before prepared : State α) : Prop where
  dout : ∀ q ∈ HostFP.grid h c, ∀ step, ∀ i : TileIndex [c.BM,c.BCS],
    row h c q i.1 < h.chunk → i.2.1.val < length h q-step*c.BCS →
    ((prepared.mem h.Dout (FPReference.dBase (HostFP.launch h c q prepared) (HostContract.parameters h)+
      (row h c q i.1*h.doutStrides.column+(step*c.BCS+i.2.1.val)*h.doutStrides.row))).read .real) =
      doutRead neg caller.dout before q.2.1 (chunk h q) (head h q) (step*c.BCS+i.2.1.val) (row h c q i.1)
  a : ∀ q ∈ HostFP.grid h c, ∀ step, ∀ i : TileIndex [c.BCS,c.BN],
    i.1.val < length h q-step*c.BCS → column h c q i.2.1 < h.K →
    ((prepared.mem h.A (FPReference.aBase (HostFP.launch h c q prepared) (HostContract.parameters h)+
      ((step*c.BCS+i.1.val)*h.aStrides.sequence+column h c q i.2.1*h.aStrides.feature))).read .real) =
      tensorRead neg caller.a before ⟨q.2.1,chunk h q*h.chunk+step*c.BCS+i.1.val,head h q,column h c q i.2.1⟩
  residual : ∀ q ∈ HostFP.grid h c, ∀ i,
    FPReference.active (HostFP.launch h c q prepared) c.BM c.BN (HostContract.parameters h) i →
    h.hasResidual = Bool.true →
    ((prepared.mem h.Residual (FPReference.offset (HostFP.launch h c q prepared) c.BM c.BN
      (HostContract.parameters h) (h.resStride h.residualStrides.batch) (h.resStride h.residualStrides.sequence)
      (h.resStride (h.headStride h.residualStrides.head)) (h.resStride h.residualStrides.feature) i)).read .fp32) =
      tensorRead neg caller.residual before (coord h c q i)

theorem expected_of_pins (M : Algebra α) (D : DotPrimitive α) (neg : α → α)
    (h : HostCall) (c : Tuning) (caller : Caller) (before prepared : State α)
    (pins : InputPins neg h c caller before prepared) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c)
    (i : TileIndex [c.BM,c.BN])
    (hi : FPReference.active (HostFP.launch h c q prepared) c.BM c.BN (HostContract.parameters h) i) :
    FPReference.expected M D c.BM c.BN c.BCS (HostContract.parameters h)
      (HostFP.launch h c q prepared) i = expected M D neg h c caller before q i := by
  have hl (step : Nat) : FPReference.leftBlock M c.BM c.BN c.BCS (HostContract.parameters h)
      (HostFP.launch h c q prepared) step = leftBlock M neg h c caller before q step := by
    funext j
    change M.cast _ _ _ (if row h c q j.1 < h.chunk ∧ j.2.1.val < length h q-step*c.BCS then _ else _) = _
    unfold leftBlock
    split_ifs with hj
    · exact congrArg _ (pins.dout q hq step j hj.1 hj.2)
    · rfl
  have hr (step : Nat) : FPReference.rightBlock M c.BN c.BCS (HostContract.parameters h)
      (HostFP.launch h c q prepared) step = rightBlock M neg h c caller before q step := by
    funext j
    change M.cast _ _ _ (if j.1.val < length h q-step*c.BCS ∧ column h c q j.2.1 < h.K then _ else _) = _
    unfold rightBlock
    split_ifs with hj
    · exact congrArg _ (pins.a q hq step j hj.1 hj.2)
    · rfl
  have hc (step : Nat) : FPReference.carry M D c.BM c.BN c.BCS (HostContract.parameters h)
      (HostFP.launch h c q prepared) step = carry M D neg h c caller before q step := by
    induction step with
    | zero => rfl
    | succ n ih => simp only [FPReference.carry,carry,ih,hl,hr]
  unfold FPReference.expected expected
  rw [hc]
  by_cases hres : h.hasResidual = Bool.true
  · simpa only [HostContract.parameters,hres,↓reduceIte] using
      congrArg (fun x => M.cast (some .fp32) .real .fp32
        (M.binary (some .fp32) .real .add
          (carry M D neg h c caller before q ((length h q+c.BCS-1)/c.BCS) i)
          (M.cast (some .fp32) .fp32 .real x))) (pins.residual q hq i hi hres)
  · simp only [HostContract.parameters,hres,↓reduceIte]
    rfl

#specNonCircular expected avoiding [HostFP.run,FP.Structural.exec,FPDefault.source,FPReference.expected]
#axiomsClean expected_of_pins
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPLogical

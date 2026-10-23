import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.TypedBinding

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RealBinding
open VeriTile.Triton FP.Structural Checked
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def inputRole (i : Fin 4) : Fin 5 := ⟨i.val,by omega⟩
def inputKey (key : Bool) : Fin 4 := if key then 1 else 0
def coefficientKey (sine : Bool) : Fin 4 := if sine then 3 else 2

theorem input_present (r : Request) (i : Fin 4) :
    present r (TypedBinding.payloadRole (inputRole i))=Bool.true := by
  simp [present,TypedBinding.payloadRole,inputRole,i.isLt]

/-- Logical input elements of actual Tensor objects, interpreted by an
explicit total word-to-Real map. No numerical arithmetic law is assumed. -/
def logical (c : Call) (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ)
    (m : Physical.Bytes) (i : Fin 4) (o : Nat) : ℝ :=
  decode (TypedBinding.logical c objects m (inputRole i) (input_present c.request i) o)

theorem direct (c : Call) (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ)
    (m : Physical.Bytes) (accepted : check c (Physical.indices c.request m)=Bool.true)
    (i : Fin 4) (o : Nat) (bound : o<(c.tensor (TypedBinding.payloadRole (inputRole i))).shape.prod) :
    decode (Physical.load m (.other c.request.device)
      (base c.request (TypedBinding.payloadRole (inputRole i))+o))=logical c objects decode m i o :=
  congrArg decode (TypedBinding.direct c objects m accepted (inputRole i) (input_present c.request i) o bound)

/-- Construct the source Real memory from current caller bytes. Signed cache
metadata is observed directly and is not passed through the payload map. -/
def state (r : Request) (decode : Native.Word → ℝ) (m : Physical.Bytes)
    (surrounding : State ℝ) : State ℝ :=
  { surrounding with mem := fun region o =>
      if region="Q" then .mk .real (decode (Physical.load m (.other r.device) (r.qBase+o)))
      else if region="K" then .mk .real (decode (Physical.load m (.other r.device) (r.kBase+o)))
      else if region="Cos" then .mk .real (decode (Physical.load m (.other r.device) (r.cosBase+o)))
      else if region="Sin" then .mk .real (decode (Physical.load m (.other r.device) (r.sinBase+o)))
      else if region="Cache" then .mk .real (decode (Physical.load m (.other r.device) (r.cacheBase+o)))
      else if region="Tables" then .mk .int (Physical.indices r m (r.tableBase+2*o))
      else if region="Lengths" then .mk .int (Physical.indices r m (r.lengthBase+2*o))
      else surrounding.mem region o }

theorem metadata_matches (r : Request) (decode : Native.Word → ℝ) (m : Physical.Bytes)
    (surrounding : State ℝ) :
    Typed.MetadataMatches r (Physical.indices r m) (state r decode m surrounding) := by
  constructor
  · intro token _
    simp [state]
  · intro o
    simp [state]

def dataOffset (r : Request) (w : Work) (odd : Bool) : Nat :=
  (w.token*r.heads w.key+w.head)*r.width+r.dim w.pair odd

def coefficientOffset (r : Request) (w : Work) (sine : Bool) : Nat :=
  w.token*(if sine then r.sinWidth else r.cosWidth)+w.pair

/-- Independent split-half rotation of logical Tensor input elements. -/
def logicalPair (c : Call) (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ)
    (m : Physical.Bytes) (w : Work) (odd : Bool) : ℝ :=
  let a := logical c objects decode m (inputKey w.key) (dataOffset c.request w Bool.false)
  let b := logical c objects decode m (inputKey w.key) (dataOffset c.request w Bool.true)
  let cosine := logical c objects decode m 2 (coefficientOffset c.request w Bool.false)
  let sine := logical c objects decode m 3 (coefficientOffset c.request w Bool.true)
  if odd then a*sine+b*cosine else a*cosine-b*sine

theorem data_bound (c : Call) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=Bool.true) (w : Work)
    (active : c.request.Active w) (odd : Bool) :
    dataOffset c.request w odd<
      (c.tensor (TypedBinding.payloadRole (inputRole (inputKey w.key)))).shape.prod := by
  rw [(tensor_facts c _ accepted _ (input_present c.request (inputKey w.key))).1]
  have h := pair_bound _ _ _ _ (pair_bound _ _ _ _ active.1 active.2.1)
    (dim_bound c.request w active odd)
  cases hk : w.key <;> simpa [hk,dataOffset,inputKey,inputRole,TypedBinding.payloadRole,
    expectedShape,Request.heads,Nat.mul_assoc] using h

theorem coefficient_bound (c : Call) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=Bool.true) (w : Work)
    (active : c.request.Active w) (sine : Bool) :
    coefficientOffset c.request w sine<
      (c.tensor (TypedBinding.payloadRole (inputRole (coefficientKey sine)))).shape.prod := by
  rw [(tensor_facts c _ accepted _ (input_present c.request (coefficientKey sine))).1]
  have h := coefficient_range c _ accepted w active sine
  cases sine <;> simp [coefficientOffset,coefficientKey,inputRole,TypedBinding.payloadRole,
    expectedShape,wordExtent,Request.coefficient,base] at h ⊢ <;> omega

theorem u_active (c : Call) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=Bool.true) (s : State ℝ)
    (head : s.pids 0<c.request.qHeads) (key : Bool) (i : Typed.ULane (Typed.uLayout c.request))
    (live : Typed.ULive c.request s key i) : c.request.Active (TypedBinding.uWork c.request s key i) := by
  have pair : i.2.2.1.val<c.request.half := by simpa [Typed.uLayout,Typed.half_width] using live.1.2.2
  have token : s.pids 1*4+i.1.val<c.request.tokens := live.1.2.1
  refine ⟨token,?_,pair⟩
  cases key
  · exact head
  · change s.pids 0/c.request.groups<c.request.kHeads
    have groups := groups_positive c (Physical.indices c.request m) accepted
    apply (Nat.div_lt_iff_lt_mul groups.1).mpr
    simpa only [groups.2] using head

theorem c_active (r : Request) (s : State ℝ) (token : s.pids 1<r.tokens)
    (head : s.pids 0<r.qHeads) (key : Bool) (i : Typed.CLane (Typed.cLayout r))
    (live : Typed.cActive (Typed.cLayout r) s key i) : r.Active (TypedBinding.cWork r s key i) := by
  have pair : i.1.val<r.half := by simpa [Typed.cLayout,Typed.half_width] using live.2
  refine ⟨token,?_,pair⟩
  cases key
  · exact head
  · simpa [Typed.cActive,Typed.cLayout] using live.1

theorem u_value_logical (c : Call) (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ)
    (m : Physical.Bytes) (accepted : check c (Physical.indices c.request m)=Bool.true)
    (s : State ℝ) (head : s.pids 0<c.request.qHeads) (key odd : Bool)
    (i : Typed.ULane (Typed.uLayout c.request)) (live : Typed.ULive c.request s key i) :
    Typed.uValue realOps (Typed.uLayout c.request) (state c.request decode m s)
      (state c.request decode m s) key odd i=
      .mk .real (logicalPair c objects decode m (TypedBinding.uWork c.request s key i) odd) := by
  let w := TypedBinding.uWork c.request s key i
  have active : c.request.Active w := u_active c m accepted s head key i live
  have hd (b : Bool) := direct c objects decode m accepted (inputKey w.key)
    (dataOffset c.request w b) (data_bound c m accepted w active b)
  have hc (sine : Bool) := direct c objects decode m accepted (coefficientKey sine)
    (coefficientOffset c.request w sine) (coefficient_bound c m accepted w active sine)
  have hd0 := hd Bool.false
  have hd1 := hd Bool.true
  have hc0 := hc Bool.false
  have hc1 := hc Bool.true
  cases key <;> cases odd <;>
    simp [Typed.uValue,logicalPair,state,Typed.uLayout,Typed.uRegion,Typed.uOffset,
      Typed.uToken,Typed.uHead,Typed.uDim,Typed.uCoefficient,Typed.half_width,TypedBinding.uWork,
      realOps,inputKey,coefficientKey,inputRole,TypedBinding.payloadRole,base,dataOffset,coefficientOffset,
      Request.heads,Request.dim,w,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] at hd0 hd1 hc0 hc1 ⊢ <;>
    rw [hd0,hd1,hc0,hc1]

theorem c_value_logical (c : Call) (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ)
    (m : Physical.Bytes) (accepted : check c (Physical.indices c.request m)=Bool.true)
    (s : State ℝ) (token : s.pids 1<c.request.tokens) (head : s.pids 0<c.request.qHeads)
    (key odd : Bool) (i : Typed.CLane (Typed.cLayout c.request))
    (live : Typed.cActive (Typed.cLayout c.request) s key i) :
    Typed.cValue realOps (Typed.cLayout c.request) (state c.request decode m s) key odd i=
      .mk .real (logicalPair c objects decode m (TypedBinding.cWork c.request s key i) odd) := by
  let w := TypedBinding.cWork c.request s key i
  have active : c.request.Active w := c_active c.request s token head key i live
  have hd (b : Bool) := direct c objects decode m accepted (inputKey w.key)
    (dataOffset c.request w b) (data_bound c m accepted w active b)
  have hc (sine : Bool) := direct c objects decode m accepted (coefficientKey sine)
    (coefficientOffset c.request w sine) (coefficient_bound c m accepted w active sine)
  have hd0 := hd Bool.false
  have hd1 := hd Bool.true
  have hc0 := hc Bool.false
  have hc1 := hc Bool.true
  cases key <;> cases odd <;>
    simp [Typed.cValue,logicalPair,state,Typed.cLayout,Typed.cInput,Typed.cDim,
      Typed.cCoefficient,Typed.half_width,TypedBinding.cWork,realOps,inputKey,coefficientKey,
      inputRole,TypedBinding.payloadRole,base,dataOffset,coefficientOffset,Request.heads,
      Request.dim,w,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] at hd0 hd1 hc0 hc1 ⊢ <;>
    rw [hd0,hd1,hc0,hc1]

#specNonCircular logicalPair avoiding [Kernels.uncached,Kernels.cached,Typed.uncachedExecution,Typed.cachedExecution,Typed.uValue,Typed.cValue]
#axiomsClean direct
#axiomsClean metadata_matches
#axiomsClean u_value_logical
#axiomsClean c_value_logical
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RealBinding

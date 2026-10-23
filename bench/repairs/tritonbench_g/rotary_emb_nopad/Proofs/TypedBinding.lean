import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.TypedNative
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.TypedBinding
open VeriTile.Triton FP.Structural
open Checked
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def payloadRole (i : Fin 5) : Fin 7 := ⟨i.val,by omega⟩

/-- Actual payload Tensor objects. The absent optional cache needs no
observations; int64 metadata is bound separately to current caller bytes. -/
structure Objects (c : Call) where
  tensor : Fin 5 → VeriTile.Torch.FP32.Tensor
  laws : ∀ i,present c.request (payloadRole i)=Bool.true → VeriTile.Torch.FP32.TensorLaws (tensor i)
  shape : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).metadata.shape=(c.tensor (payloadRole i)).shape
  object : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).object=(c.tensor (payloadRole i)).object
  pointer : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).dataPtr.toNat=(c.tensor (payloadRole i)).base
  device : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).metadata.device=⟨"cuda",c.request.device⟩
  contiguous : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).metadata.isContiguous=(c.tensor (payloadRole i)).contiguous
  negative : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).metadata.isNeg=(c.tensor (payloadRole i)).negative
  storageOffset : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).storageOffset=(c.tensor (payloadRole i)).storageOffset
  backing : ∀ i,present c.request (payloadRole i)=Bool.true →
    (tensor i).storage.base=(c.tensor (payloadRole i)).allocationBase ∧
    4*(tensor i).storage.elements=(c.tensor (payloadRole i)).allocationBytes

def read (r : Request) (m : Physical.Bytes) (address : Nat) : VeriTile.Torch.FP32.Word :=
  ⟨VeriTile.Bench.Repairs.RotaryEmb.Physical.decode (fun i => m (.other r.device) (address+i.val))⟩

def logical (c : Call) (objects : Objects c) (m : Physical.Bytes) (i : Fin 5)
    (pi : present c.request (payloadRole i)=Bool.true) (o : Nat) : Native.Word :=
  (VeriTile.Torch.FP32.logicalValue (read c.request m) (objects.tensor i) (objects.laws i pi) o).bits

theorem direct (c : Call) (objects : Objects c) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=Bool.true) (i : Fin 5)
    (pi : present c.request (payloadRole i)=Bool.true) (o : Nat)
    (bound : o<(c.tensor (payloadRole i)).shape.prod) :
    Physical.load m (.other c.request.device) (base c.request (payloadRole i)+o)=
      logical c objects m i pi o := by
  have checked := tensor_checked c _ accepted
  simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at checked
  have hi := checked.1 (payloadRole i) pi
  have hc : (objects.tensor i).metadata.isContiguous=Bool.true :=
    (objects.contiguous i pi).trans hi.2.2.2.2.1
  have hn : (objects.tensor i).metadata.isNeg=Bool.false :=
    (objects.negative i pi).trans hi.2.2.2.2.2.2.1
  have size : (objects.tensor i).metadata.numel=(c.tensor (payloadRole i)).shape.prod := by
    simp [VeriTile.Torch.TensorMetadata.numel,objects.shape i pi]
  have facts := tensor_facts c _ accepted (payloadRole i) pi
  have pointer := (facts.2.2.resolve_left (by omega)).2.1
  have hp := congrArg BitVec.toNat
    (VeriTile.Torch.FP32.actual_pointer (objects.tensor i) (objects.laws i pi) hc (by omega))
  have span := (VeriTile.Torch.FP32.dataView (objects.tensor i) (objects.laws i pi) hc).spanFits
  have baseBound : (VeriTile.Torch.FP32.dataView (objects.tensor i) (objects.laws i pi) hc).base<2^64 := by
    change _+4*(objects.tensor i).metadata.numel≤2^64 at span
    omega
  simp only [VeriTile.Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero,
    BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound,objects.pointer i pi] at hp
  have eq := VeriTile.Torch.FP32.direct_read_eq_logical (read c.request m)
    (objects.tensor i) (objects.laws i pi) hc hn o (by omega)
  simpa only [logical,VeriTile.Torch.FP32.DataView.address,hp,pointer,read,
    Physical.load,VeriTile.Bench.Repairs.RotaryEmb.Physical.load,Nat.mul_add,Nat.add_assoc] using
      congrArg (fun w : VeriTile.Torch.FP32.Word => w.bits) eq

def state (r : Request) (m : Physical.Bytes) (surrounding : State Native.Node) : State Native.Node :=
  { surrounding with mem := fun region o =>
      if region="Q" then .mk .real (Native.atom (Physical.load m (.other r.device) (r.qBase+o)))
      else if region="K" then .mk .real (Native.atom (Physical.load m (.other r.device) (r.kBase+o)))
      else if region="Cos" then .mk .real (Native.atom (Physical.load m (.other r.device) (r.cosBase+o)))
      else if region="Sin" then .mk .real (Native.atom (Physical.load m (.other r.device) (r.sinBase+o)))
      else if region="Cache" then .mk .real (Native.atom (Physical.load m (.other r.device) (r.cacheBase+o)))
      else if region="Tables" then .mk .int (Physical.indices r m (r.tableBase+2*o))
      else if region="Lengths" then .mk .int (Physical.indices r m (r.lengthBase+2*o))
      else surrounding.mem region o }

theorem metadata_matches (r : Request) (m : Physical.Bytes) (surrounding : State Native.Node) :
    Typed.MetadataMatches r (Physical.indices r m) (state r m surrounding) := by
  constructor
  · intro token _
    simp [state]
  · intro o
    simp [state]

def uWork (r : Request) (s : State A) (key : Bool) (i : Typed.ULane (Typed.uLayout r)) : Work :=
  ⟨key,s.pids 1*4+i.1.val,if key then s.pids 0/r.groups else s.pids 0,i.2.2.1.val⟩

def cWork (r : Request) (s : State A) (key : Bool) (i : Typed.CLane (Typed.cLayout r)) : Work :=
  ⟨key,s.pids 1,s.pids 0,i.1.val⟩

theorem u_graph (p : Native.Instructions) (r : Request) (uncached : r.cached=Bool.false)
    (m : Physical.Bytes) (s : State Native.Node) (key odd : Bool) (i : Typed.ULane (Typed.uLayout r)) :
    Typed.uGraphValue p r (state r m s) key odd i=.mk .real
      (Native.atom (Native.expected p r (uWork r s key i) (Physical.load m) odd)) := by
  cases key <;> cases odd <;>
    simp [Typed.uGraphValue,Typed.splitGraph,Typed.uLayout,Typed.uRegion,Typed.uOffset,
      Typed.uToken,Typed.uHead,Typed.uDim,Typed.uCoefficient,Typed.half_width,
      state,Native.expected,Native.atom,VeriTile.Bench.Repairs.RotaryEmb.Native.atom,
      uWork,Request.data,Request.heads,Request.coefficient,Request.dim,uncached,
      Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

theorem c_graph (p : Native.Instructions) (r : Request) (cached : r.cached=Bool.true)
    (m : Physical.Bytes) (s : State Native.Node) (key odd : Bool) (i : Typed.CLane (Typed.cLayout r)) :
    Typed.cGraphValue p r (state r m s) key odd i=.mk .real
      (Native.atom (Native.expected p r (cWork r s key i) (Physical.load m) odd)) := by
  cases key <;> cases odd <;>
    simp [Typed.cGraphValue,Typed.splitGraph,Typed.cLayout,Typed.cInput,Typed.cDim,
      Typed.cCoefficient,Typed.half_width,state,Native.expected,Native.atom,
      VeriTile.Bench.Repairs.RotaryEmb.Native.atom,cWork,Request.data,Request.heads,
      Request.coefficient,Request.dim,cached,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

theorem u_active (c : Call) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=Bool.true) (s : State Native.Node)
    (head : s.pids 0<c.request.qHeads) (key : Bool) (i : Typed.ULane (Typed.uLayout c.request))
    (live : Typed.ULive c.request s key i) : c.request.Active (uWork c.request s key i) := by
  have pair : i.2.2.1.val<c.request.half := by
    simpa [Typed.uLayout,Typed.half_width] using live.1.2.2
  have token : s.pids 1*4+i.1.val<c.request.tokens := live.1.2.1
  refine ⟨token,?_,pair⟩
  cases key
  · exact head
  · change s.pids 0/c.request.groups<c.request.kHeads
    have groups := groups_positive c (Physical.indices c.request m) accepted
    apply (Nat.div_lt_iff_lt_mul groups.1).mpr
    simpa only [groups.2] using head

theorem c_active (r : Request) (s : State Native.Node) (token : s.pids 1<r.tokens)
    (head : s.pids 0<r.qHeads) (key : Bool) (i : Typed.CLane (Typed.cLayout r))
    (live : Typed.cActive (Typed.cLayout r) s key i) : r.Active (cWork r s key i) := by
  have pair : i.1.val<r.half := by simpa [Typed.cLayout,Typed.half_width] using live.2
  refine ⟨token,?_,pair⟩
  cases key
  · exact head
  · simpa [Typed.cActive,Typed.cLayout] using live.1

theorem positions (c : Call) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=Bool.true) (cached : c.request.cached=Bool.true)
    (s : State Native.Node) (token : s.pids 1<c.request.tokens) (nonempty : 0<c.request.half) :
    (Typed.cPast (Typed.cLayout c.request) (state c.request m s)).toNat=
      c.request.past (Physical.indices c.request m) (s.pids 1) ∧
    (Typed.cBlock (Typed.cLayout c.request) (state c.request m s)).toNat=
      c.request.block (Physical.indices c.request m) (s.pids 1) := by
  have hp : (Typed.cPast (Typed.cLayout c.request) (state c.request m s)).toNat=
      c.request.past (Physical.indices c.request m) (s.pids 1) := by
    simp [Typed.cPast,Typed.cLayout,state,Request.past]
  have nonneg := Typed.metadata_nonnegative c (Physical.indices c.request m) accepted cached
    (state c.request m s) (metadata_matches _ _ _) token nonempty
  refine ⟨hp,?_⟩
  unfold Typed.cBlock
  rw [Int.tdiv_eq_ediv_of_nonneg nonneg.1,←Int.toNat_of_nonneg nonneg.1,hp]
  simp only [Typed.cLayout,Nat.cast_one,mul_one,←Int.natCast_ediv,Int.toNat_natCast]
  simp [state,Request.block]

/-- The signed typed page/slot expressions denote the checked caller cache
destination. This binds cache placement independently of the rotation value. -/
theorem c_destination (c : Call) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=Bool.true) (cached : c.request.cached=Bool.true)
    (s : State Native.Node) (token : s.pids 1<c.request.tokens) (nonempty : 0<c.request.half)
    (key odd : Bool) (i : Typed.CLane (Typed.cLayout c.request)) :
    base c.request (destinationRole c.request key)+
      Typed.cOffset (Typed.cLayout c.request) (state c.request m s) key odd i=
      destination c.request (Physical.indices c.request m) (cWork c.request s key i) odd := by
  cases key
  · simp [base,destinationRole,role,Typed.cOffset,Typed.cInput,Typed.cDim,Typed.cLayout,
      Typed.half_width,state,destination,Request.destination,Request.data,Request.heads,
      cWork,Request.dim,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]
  · have nonneg := Typed.metadata_nonnegative c (Physical.indices c.request m) accepted cached
      (state c.request m s) (metadata_matches _ _ _) token nonempty
    rw [Typed.c_offset c.request _ nonneg.1 nonneg.2]
    rw [(positions c m accepted cached s token nonempty).1,(positions c m accepted cached s token nonempty).2]
    simp [base,destinationRole,cached,destination,Request.destination,Request.slot,cWork,
      state,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

theorem u_physical (rest : Algebra Native.Node) (p : Native.Instructions) (c : Call)
    (m : Physical.Bytes) (accepted : check c (Physical.indices c.request m)=Bool.true)
    (uncached : c.request.cached=Bool.false) (s : State Native.Node) (head : s.pids 0<c.request.qHeads)
    (key odd : Bool) (i : Typed.ULane (Typed.uLayout c.request)) (live : Typed.ULive c.request s key i) :
    Physical.load (Physical.native p c m) (.other c.request.device)
      (destination c.request (Physical.indices c.request m) (uWork c.request s key i) odd)=
      (((Typed.uncachedExecution rest (Native.ops p Bool.false) (Typed.uLayout c.request)
        (state c.request m s)).val.mem (Typed.uRegion (Typed.uLayout c.request) key)
          (Typed.uOffset (Typed.uLayout c.request) s key odd i)).read .real).value := by
  rw [Physical.native_result p c m accepted _ (u_active c m accepted s head key i live),
    Typed.uncached_memory,Typed.u_reference_target]
  have result := Typed.u_readback (Native.ops p Bool.false) c (Physical.indices c.request m) accepted
    (state c.request m s) key odd i live
  change (Typed.uTarget (Native.ops p Bool.false) c.request (state c.request m s)).mem
    (Typed.uRegion (Typed.uLayout c.request) key) (Typed.uOffset (Typed.uLayout c.request) s key odd i)=_ at result
  rw [result,Typed.u_value_graph,u_graph p c.request uncached m s]
  rfl

theorem c_physical (rest : Algebra Native.Node) (p : Native.Instructions) (c : Call)
    (m : Physical.Bytes) (accepted : check c (Physical.indices c.request m)=Bool.true)
    (cached : c.request.cached=Bool.true) (s : State Native.Node) (token : s.pids 1<c.request.tokens)
    (nonempty : 0<c.request.half) (head : s.pids 0<c.request.qHeads)
    (key odd : Bool) (i : Typed.CLane (Typed.cLayout c.request))
    (live : Typed.cActive (Typed.cLayout c.request) s key i) :
    Physical.load (Physical.native p c m) (.other c.request.device)
      (destination c.request (Physical.indices c.request m) (cWork c.request s key i) odd)=
      (((Typed.cachedExecution rest (Native.ops p Bool.true) (Typed.cLayout c.request)
        (state c.request m s)).val.mem (Typed.cRegion (Typed.cLayout c.request) key)
          (Typed.cOffset (Typed.cLayout c.request) (state c.request m s) key odd i)).read .real).value := by
  rw [Physical.native_result p c m accepted _ (c_active c.request s token head key i live),
    Typed.cached_memory rest _ _ _ token]
  have nonneg := Typed.metadata_nonnegative c (Physical.indices c.request m) accepted cached
    (state c.request m s) (metadata_matches _ _ _) token nonempty
  rw [Typed.c_readback _ _ _ nonneg.1 nonneg.2 head key odd i live,Typed.c_value_graph,c_graph p c.request cached m s]
  rfl

#axiomsClean direct
#axiomsClean metadata_matches
#axiomsClean u_graph
#axiomsClean c_graph
#axiomsClean positions
#axiomsClean c_destination
#axiomsClean u_physical
#axiomsClean c_physical
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.TypedBinding

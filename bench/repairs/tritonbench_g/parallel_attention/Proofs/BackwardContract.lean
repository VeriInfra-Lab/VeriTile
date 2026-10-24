import bench.repairs.tritonbench_g.parallel_attention.Proofs.Pointer

namespace VeriTile.Bench.Repairs.ParallelAttention.Checked
open HostAPI
set_option maxHeartbeats 1000000

/-- A rank-three dZ has an irrelevant singleton feature coordinate. -/
def cotangentLift (z : Bool) (t : TensorView) : TensorView :=
  if z then {t with strides:=t.strides++[0]} else t

def cotangentWidth (s : Shape) (z : Bool) : Nat := if z then 1 else s.V

def cotangentShape (s : Shape) (z : Bool) : List Nat :=
  if z then [s.B,s.H,s.T] else [s.B,s.H,s.T,s.V]

/-- Raw physical reads are logical reads only after the lazy flags are rejected. -/
def CotangentKind (t : TensorView) : Prop :=
  t.tensor.layout=.strided ∧ t.tensor.dtype=.floating .fp32 ∧
  t.tensor.device.kind="cuda" ∧ t.tensor.isNeg=Bool.false ∧ t.isConj=Bool.false
instance (t : TensorView) : Decidable (CotangentKind t) := by unfold CotangentKind;infer_instance

def CotangentStrides (z : Bool) (t : TensorView) : Prop :=
  match (cotangentLift z t).strides with
  | [sb,sh,st,se] => 0 ≤ sb ∧ 0 ≤ sh ∧ 0 ≤ st ∧ 0 ≤ se
  | _ => False
instance (z : Bool) (t : TensorView) : Decidable (CotangentStrides z t) := by
  unfold CotangentStrides;split <;> infer_instance

def cotangentView (z : Bool) (t : TensorView) : View :=
  match (cotangentLift z t).strides with
  | [sb,sh,st,se] => ⟨t.pointer/4,sb.toNat,sh.toNat,st.toNat,se.toNat⟩
  | _ => ⟨t.pointer/4,0,0,0,0⟩

/-- The exact nonnegative-stride storage envelope checked before copying. -/
def cotangentSpan (s : Shape) (z : Bool) (t : TensorView) : Nat :=
  let v:=cotangentView z t
  (s.B-1)*v.batchStride+(s.H-1)*v.headStride+(s.T-1)*v.timeStride+
    (cotangentWidth s z-1)*v.featureStride+1

def CotangentStored (s : Shape) (z : Bool) (t : TensorView) : Prop :=
  Machine.Empty s ∨ Stored t (cotangentSpan s z t)
instance (s : Shape) (z : Bool) (t : TensorView) : Decidable (CotangentStored s z t) := by
  unfold CotangentStored Machine.Empty;infer_instance

def IncomingCotangent (c : Call) (z : Bool) (t : TensorView) : Prop :=
  CotangentKind t ∧ t.tensor.device=c.q.tensor.device ∧
  t.tensor.shape=cotangentShape c.shape z ∧ CotangentStrides z t ∧ CotangentStored c.shape z t
instance (c : Call) (z : Bool) (t : TensorView) : Decidable (IncomingCotangent c z t) := by
  unfold IncomingCotangent;infer_instance

def PreparedCotangent (c : Call) (z : Bool) (t : TensorView) : Prop :=
  IncomingCotangent c z t ∧ t.tensor.isContiguous=Bool.true ∧
  DenseStrides c.shape (cotangentWidth c.shape z) (cotangentLift z t) ∧
  (Machine.Empty c.shape ∨ t.pointer+4*Machine.formedLimit c.shape<2^63)
instance (c : Call) (z : Bool) (t : TensorView) : Decidable (PreparedCotangent c z t) := by
  unfold PreparedCotangent Machine.Empty;infer_instance

/-- Classification of ordinary Tensor objects precedes their metadata queries.
The query ABI supplies exactly one stride per dimension. -/
inductive CotangentObjects where
  | nonTensor
  | query (do_ dz : TensorView)

def backwardCheck (c : Call) : CotangentObjects → Except Error Unit
  | .nonTensor => .error .object
  | .query do_ dz =>
    if checkCall c≠.ok () then .error .metadata
    else if ¬IncomingCotangent c Bool.false do_ ∨ ¬IncomingCotangent c Bool.true dz then .error .metadata
    else .ok ()

def BackwardAdmitted (c : Call) : CotangentObjects → Prop
  | .nonTensor => False
  | .query do_ dz => Domain c ∧ IncomingCotangent c Bool.false do_ ∧ IncomingCotangent c Bool.true dz

def preparedCheck (c : Call) (do_ dz : TensorView) : Except Error Unit :=
  if checkCall c≠.ok () then .error .metadata
  else if ¬PreparedCotangent c Bool.false do_ ∨ ¬PreparedCotangent c Bool.true dz then .error .storage
  else .ok ()

theorem backwardCheck_iff (c : Call) (xs : CotangentObjects) :
    backwardCheck c xs=.ok () ↔ BackwardAdmitted c xs := by
  cases xs with
  | nonTensor => simp [backwardCheck,BackwardAdmitted]
  | query do_ dz =>
    simp only [backwardCheck,BackwardAdmitted]
    split_ifs <;> simp_all [checkCall_iff] <;> aesop

theorem backward_rejected_iff (c : Call) (xs : CotangentObjects) :
    (∃ e,backwardCheck c xs=.error e) ↔ ¬BackwardAdmitted c xs := by
  rw [←backwardCheck_iff];cases backwardCheck c xs <;> simp

theorem preparedCheck_iff (c : Call) (do_ dz : TensorView) :
    preparedCheck c do_ dz=.ok () ↔
      Domain c ∧ PreparedCotangent c Bool.false do_ ∧ PreparedCotangent c Bool.true dz := by
  unfold preparedCheck
  split_ifs <;> simp_all [checkCall_iff] <;> aesop

/-- Checked metadata and actual byte pointers for the saved inputs and the two
prepared kernel arguments. Empty backward skips the preparation phase. -/
structure PreparedBinding (s : Shape) (x : HostAPI.Inputs) (do_ dz : Tensor) where
  call : Call
  domain : Domain call
  shape : call.shape=s
  saved : inputs call=x
  incomingOutput : TensorView
  incomingNormalizer : TensorView
  incomingChecked : IncomingCotangent call Bool.false incomingOutput ∧
    IncomingCotangent call Bool.true incomingNormalizer
  output : TensorView
  normalizer : TensorView
  checked : Machine.Empty s ∨
    (PreparedCotangent call Bool.false output ∧ PreparedCotangent call Bool.true normalizer)
  outputHandle : do_=tensor s .forward (output.pointer/4)
  normalizerHandle : dz=tensor s .normalizer (normalizer.pointer/4)

/-- The pre-copy guard additionally binds incoming views to actual raw storage.
The separate Materialized premises describe only ordinary contiguous copying. -/
structure BackwardBinding (s : Shape) (x : HostAPI.Inputs)
    (doView dzView : View) (do_ dz : Tensor) extends PreparedBinding s x do_ dz where
  outputView : doView=cotangentView Bool.false incomingOutput
  normalizerView : dzView=cotangentView Bool.true incomingNormalizer

theorem incoming_resolved (c : Call) (z : Bool) (t : TensorView) (h : IncomingCotangent c z t) :
    t.tensor.isNeg=Bool.false ∧ t.isConj=Bool.false := ⟨h.1.2.2.2.1,h.1.2.2.2.2⟩

theorem cotangent_view_base (z : Bool) (t : TensorView) :
    (cotangentView z t).base=t.pointer/4 := by unfold cotangentView;split <;> rfl

theorem incoming_lane (c : Call) (z : Bool) (t : TensorView) (h : IncomingCotangent c z t)
    (bh row e : Nat) (hb : bh<c.shape.B*c.shape.H) (hr : row<c.shape.T)
    (he : e<cotangentWidth c.shape z) (byte : Fin 4) :
    let address:=4*((cotangentView z t).address c.shape bh row e)+byte.val
    t.storageBase≤address ∧ address<t.storageBase+t.capacity := by
  have hh : 0<c.shape.H := by nlinarith
  have bbound : bh/c.shape.H<c.shape.B := (Nat.div_lt_iff_lt_mul hh).mpr hb
  have hbound : bh%c.shape.H<c.shape.H := Nat.mod_lt _ hh
  have bp : 0<c.shape.B := by nlinarith
  have ne : ¬Machine.Empty c.shape := by
    simp only [Machine.Empty,ne_of_gt bp,ne_of_gt hh,ne_of_gt (Nat.zero_lt_of_lt hr),or_self,not_false_eq_true]
  have stored : Stored t (cotangentSpan c.shape z t) := h.2.2.2.2.resolve_left ne
  let v:=cotangentView z t
  let offset:=(bh/c.shape.H)*v.batchStride+(bh%c.shape.H)*v.headStride+row*v.timeStride+e*v.featureStride
  have lo0:=Nat.mul_le_mul_right v.batchStride (show bh/c.shape.H≤c.shape.B-1 by omega)
  have lo1:=Nat.mul_le_mul_right v.headStride (show bh%c.shape.H≤c.shape.H-1 by omega)
  have lo2:=Nat.mul_le_mul_right v.timeStride (show row≤c.shape.T-1 by omega)
  have lo3:=Nat.mul_le_mul_right v.featureStride (show e≤cotangentWidth c.shape z-1 by omega)
  have bound : offset<cotangentSpan c.shape z t := by unfold offset cotangentSpan;dsimp [v] at *;omega
  obtain ⟨eq,lo,hi⟩:=stored_address t _ offset stored bound
  have base:=cotangent_view_base z t
  change t.storageBase≤4*(v.address c.shape bh row e)+byte.val ∧
    4*(v.address c.shape bh row e)+byte.val<t.storageBase+t.capacity
  have addr : v.address c.shape bh row e=t.pointer/4+offset := by
    dsimp only [View.address,offset];rw [show v.base=t.pointer/4 from base];omega
  rw [addr,eq]
  have byteBound:=byte.isLt
  omega

/-- Natural-stride word views and the queried signed Torch strides have the
same active address; rejection of negative strides makes this conversion exact. -/
theorem cotangent_stride_sum (z : Bool) (t : TensorView) (hs : CotangentStrides z t)
    (b h row e : Nat) :
    ((b*(cotangentView z t).batchStride+h*(cotangentView z t).headStride+
      row*(cotangentView z t).timeStride+e*(cotangentView z t).featureStride : Nat) : Int)=
      storageOffset (cotangentLift z t) b h row e := by
  unfold CotangentStrides at hs
  unfold cotangentView storageOffset
  split at hs <;> simp_all only
  rcases hs with ⟨hb,hh,hr,he⟩
  simp only [Nat.cast_add,Nat.cast_mul,Int.toNat_of_nonneg hb,Int.toNat_of_nonneg hh,
    Int.toNat_of_nonneg hr,Int.toNat_of_nonneg he]

theorem prepared_view_address (c : Call) (z : Bool) (t : TensorView)
    (h : PreparedCotangent c z t) (bh row e : Nat)
    (hb : bh<c.shape.B*c.shape.H) (hr : row<c.shape.T) (he : e<cotangentWidth c.shape z) :
    (cotangentView z t).address c.shape bh row e=
      t.pointer/4+dense c.shape.T (cotangentWidth c.shape z) bh row e := by
  have hh : 0<c.shape.H := by nlinarith
  have bb : bh/c.shape.H<c.shape.B := (Nat.div_lt_iff_lt_mul hh).mpr hb
  have hb' : bh%c.shape.H<c.shape.H := Nat.mod_lt _ hh
  have actual:=dense_actual_strides c.shape (cotangentWidth c.shape z) (cotangentLift z t)
    h.2.2.1 (bh/c.shape.H) (bh%c.shape.H) row e bb hb' hr he
  rw [Nat.mul_comm (bh/c.shape.H) c.shape.H,Nat.div_add_mod] at actual
  have offsets:=cotangent_stride_sum z t h.1.2.2.2.1 (bh/c.shape.H) (bh%c.shape.H) row e
  rw [actual] at offsets
  have offsets':=(Int.natCast_inj).mp offsets
  unfold View.address
  rw [cotangent_view_base]
  omega

theorem prepared_pointer_room (c : Call) (z : Bool) (t : TensorView)
    (h : PreparedCotangent c z t) (active : Machine.Active c.shape)
    (offset : Nat) (bound : offset<Machine.formedLimit c.shape) :
    t.pointer+4*offset<2^63 := by
  have room:=h.2.2.2
  rcases room with empty|room
  · unfold Machine.Empty Machine.Active at *;omega
  · omega

#axiomsClean backwardCheck_iff
#axiomsClean backward_rejected_iff
#axiomsClean preparedCheck_iff
#axiomsClean incoming_lane
#axiomsClean prepared_pointer_room
#axiomsClean cotangent_stride_sum
#axiomsClean prepared_view_address
#specNonCircular IncomingCotangent avoiding [combinedRun,HostAPI.forward,HostAPI.backward]
#specNonCircular PreparedCotangent avoiding [combinedRun,HostAPI.forward,HostAPI.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ParallelAttention.Checked

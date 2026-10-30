import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Metadata
namespace VeriTile.Bench.Repairs.RotaryTransformOps.Host
open Native Physical

def size (s : Shape) := s.total*s.heads*s.width
def copied (s : Shape) (r : Region) (a : Nat) : Prop :=
  s.inplace=false ∧ r=.output ∧ a<size s ∧ 2*s.half≤a%s.width
instance (s : Shape) (r : Region) (a : Nat) : Decidable (copied s r a) := inferInstanceAs (Decidable (_∧_∧_∧_))
/-- The host copies only the non-rotary suffix into the fresh output. The
rotary prefix remains the allocator's arbitrary initial bytes. -/
def prepare {α : Type} (s : Shape) (m : Memory α) : Memory α := fun r a =>
  if copied s r a then m .input a else m r a
def prepareBytes (s : Shape) (m : Bytes) : Bytes := fun r b =>
  if copied s r (b/4) then m .input b else m r b

theorem prepare_load (s : Shape) (m : Bytes) : load (prepareBytes s m)=prepare s (load m) := by
  funext r a
  have hd (i : Fin 4) : (4*a+i.val)/4=a := by omega
  simp only [load,prepareBytes,hd,prepare]
  split_ifs <;> rfl

theorem prepare_readonly {α : Type} (s : Shape) (m : Memory α) (r : Region) (a : Nat)
    (h : r≠.output) : prepare s m r a=m r a := by simp [prepare,copied,h]
theorem prepareBytes_readonly (s : Shape) (m : Bytes) (r : Region) (a : Nat)
    (h : r≠.output) : prepareBytes s m r a=m r a := by simp [prepareBytes,copied,h]
theorem prepare_metadata (s : Shape) (a : MetadataABI) (m : Bytes) :
    metadata a (prepareBytes s m)=metadata a m := by
  simp [metadata,integer,load,prepareBytes,copied]
theorem prepared_pair (c : Config) (p : Instructions) (a : MetadataABI) (m : Bytes) (w : Work) (o : Bool) :
    pairWord c p w (load (prepareBytes c.shape (normalizeFor c.shape a m))) o=pairWord c p w (load m) o := by
  cases hv : c.shape.varlen <;> simp [pairWord,load,prepareBytes,copied,normalizeFor,hv,Physical.normalize]
theorem prepare_pair {α : Type} (c : Config) (p : Ops α) (m : Memory α) (w : Work) :
    pairValues c p w (prepare c.shape m)=pairValues c p w m := by
  simp [pairValues,prepare,copied]

def empty (s : Shape) : Bool := decide (s.total=0 ∨ s.half=0)
theorem empty_schedule (c : Config) (hc : c.Valid) (h : empty c.shape=true) : schedule c=[] := by
  have h' : c.shape.total=0 ∨ c.shape.half=0 := by simpa [empty] using h
  rcases h' with ht|hp
  · exact empty_total_schedule c hc ht
  · exact zero_half_schedule c hp

structure Call where
  shape : Shape
  abi : MetadataABI
  inputIdentity : Nat
  freshIdentity : Nat
  device : Nat
structure Returned (α : Type) where
  valueIdentity : Nat
  valueRegion : Region
  shape : Shape
  fp32 : Bool
  device : Nat
  deviceRestored : Nat
  memory : Memory α

def selected (call : Call) (callerDevice : Nat) (p : Instructions) (m : Bytes) : Returned Byte :=
  let prepared := prepareBytes call.shape (normalizeFor call.shape call.abi m)
  ⟨if call.shape.inplace then call.inputIdentity else call.freshIdentity,
   call.shape.output,call.shape,true,call.device,callerDevice,
   if empty call.shape then prepared else dynamic call.shape call.abi.unit p prepared⟩
noncomputable def mathematical (call : Call) (callerDevice : Nat) (initial : Bytes) (m : Memory ℝ) : Returned ℝ :=
  let prepared := prepare call.shape m
  ⟨if call.shape.inplace then call.inputIdentity else call.freshIdentity,
   call.shape.output,call.shape,true,call.device,callerDevice,
   if empty call.shape then prepared else source (originalConfiguration call.shape call.abi initial) realOps prepared⟩

theorem selected_memory (call : Call) (callerDevice : Nat) (p : Instructions) (m : Bytes)
    (hc : (originalConfiguration call.shape call.abi m).Valid) :
    (selected call callerDevice p m).memory=
      native (originalConfiguration call.shape call.abi m) p (prepareBytes call.shape (normalizeFor call.shape call.abi m)) := by
  have eq : configuration call.shape call.abi.unit (prepareBytes call.shape (normalizeFor call.shape call.abi m))=
      originalConfiguration call.shape call.abi m := by
    change Config.mk _ (canonicalMetadata _ (metadata _ _))=_
    rw [prepare_metadata]
    exact normalizeFor_configuration call.shape call.abi m
  unfold selected
  by_cases h : empty call.shape=true
  · simp only [h,ite_true]
    rw [native,empty_schedule _ hc h]
    rfl
  · simp only [h,Bool.false_eq_true,ite_false]
    rw [dynamic_eq,eq]
theorem mathematical_memory (call : Call) (callerDevice : Nat) (initial : Bytes) (m : Memory ℝ)
    (hc : (originalConfiguration call.shape call.abi initial).Valid) :
    (mathematical call callerDevice initial m).memory=
      source (originalConfiguration call.shape call.abi initial) realOps (prepare call.shape m) := by
  unfold mathematical
  by_cases h : empty call.shape=true
  · simp only [h,ite_true]
    rw [source_eq_run _ hc,run,empty_schedule _ hc h]
    rfl
  · simp [h]

theorem returned_tensor (call : Call) (callerDevice : Nat) (p : Instructions) (m : Bytes) :
    (selected call callerDevice p m).valueIdentity=(if call.shape.inplace then call.inputIdentity else call.freshIdentity) ∧
    (selected call callerDevice p m).valueRegion=call.shape.output ∧
    (selected call callerDevice p m).shape=call.shape ∧
    (selected call callerDevice p m).fp32=true ∧
    (selected call callerDevice p m).device=call.device ∧
    (selected call callerDevice p m).deviceRestored=callerDevice := ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩
def launch (s : Shape) : (Nat×Nat×Nat)×(Nat×Nat×Nat×Nat) :=
  ((s.rowBlocks,s.batches,s.heads),(s.blockM,s.blockK,4,3))

/-- The pointer arithmetic of the host suffix-copy view. -/
theorem address_mod (s : Shape) (n h d : Nat) (hd : d<s.width) : s.address n h d%s.width=d := by
  simp [Shape.address,Nat.add_mod,Nat.mod_eq_of_lt hd]
theorem address_bound (s : Shape) (n h d : Nat) (hn : n<s.total) (hh : h<s.heads) (hd : d<s.width) :
    s.address n h d<size s := by
  have row : n*s.heads+h<s.total*s.heads := by
    calc
      _ < n*s.heads+s.heads := Nat.add_lt_add_left hh _
      _ = (n+1)*s.heads := by ring
      _ ≤ _ := Nat.mul_le_mul_right _ (by omega)
  unfold Shape.address size
  calc
    _ < (n*s.heads+h)*s.width+s.width := Nat.add_lt_add_left hd _
    _ = (n*s.heads+h+1)*s.width := by ring
    _ ≤ _ := Nat.mul_le_mul_right _ (by omega)
theorem prepare_tail {α : Type} (s : Shape) (m : Memory α) (n h d : Nat)
    (hn : n<s.total) (hh : h<s.heads) (hd : d<s.width) (tail : 2*s.half≤d) :
    prepare s m s.output (s.address n h d)=m .input (s.address n h d) := by
  have bound := address_bound s n h d hn hh hd
  have rem := address_mod s n h d hd
  cases hi : s.inplace <;> simp [prepare,copied,Shape.output,hi,bound,rem,tail]
#axiomsClean prepare_load
#axiomsClean selected_memory
#axiomsClean mathematical_memory
#axiomsClean returned_tensor
#axiomsClean prepare_tail
end VeriTile.Bench.Repairs.RotaryTransformOps.Host

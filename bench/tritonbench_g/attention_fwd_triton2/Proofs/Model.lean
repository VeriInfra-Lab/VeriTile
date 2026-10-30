import bench.tritonbench_g.attention_fwd_triton2.Proofs.Inputs

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000

def roleOf (r : RegionName) : Option Role :=
  if r=tag .query then some .query else if r=tag .key then some .key
  else if r=tag .value then some .value else if r=tag .queryScale then some .queryScale
  else if r=tag .keyScale then some .keyScale else if r=tag .output then some .output else none
/-- All input views share one physical reader, including legal read aliases. -/
def caller {α : Type} [Inhabited α] (c : Call) (read : Nat→α) (r : RegionName) (off : Nat) : α :=
  match roleOf r with | some role=>read (base c role+byteWidth role*off) | none=>default
theorem caller_tag {α : Type} [Inhabited α] (c : Call) (read : Nat→α) (r : Role) (off : Nat) :
    caller c read (tag r) off=read (base c r+byteWidth r*off) := by
  cases r <;> simp [caller,roleOf,tag]
def entry {α : Type} [Inhabited α] (read : RegionName→Nat→α) : State α where
  mem:=fun r off=>if r=tag .output then .mk .bf16 (read r off) else .mk .real (read r off)
  regs:=fun _ _ _=>none
  pids:=fun _=>0
  numPids:=fun _=>1
  undef:=fun d _ _=>defaultValue d

def Exterior (c : Call) (r : RegionName) (off : Nat) : Prop :=
  r≠tag .output ∨ ∀ b<(parameters c).batches,∀ h<(parameters c).heads,
    ∀ n<(parameters c).length,∀ d<96,off≠offset (c .output).strides b h n d
def RawFrame {α : Type} (c : Call) (before after : State α) : Prop :=
  ∀ r off,Exterior c r off → after.mem r off=before.mem r off

theorem exterior_outside {α : Type} (c : Call) (before : State α) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) (r : RegionName) (off : Nat)
    (outside : Exterior c r off) : Outside (parameters c) (grid (parameters c)) work before r off := by
  rcases outside with hr|ho
  · exact Or.inl hr
  · right
    intro a ha i active
    rcases a with ⟨x,y,z⟩
    have member:=(mem_programs _ _ _ _).mp (order.mem_iff.mpr ha)
    have head:=head_decode (parameters c) y member.2.1
    change x*128+i.1.val<(parameters c).length ∧ i.2.1.val<96 at active
    have outside:=ho (y/(parameters c).heads) head.2.1 (y%(parameters c).heads) head.2.2
      (x*128+i.1.val) active.1 i.2.1.val active.2
    simpa [Source.address,Source.base,start,ids,offset,parameters] using outside

/-- Derive complete quoted-grid success from admission, including every
logical output and every untouched source location. No run/output premise. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (c : Call) (hc : InputContract c) (laws : TensorLaws c) (before : State α)
    (work : List Program) (order : (programs (grid (parameters c))).Perm work) :
    ∃ after,runPrograms (fun _=>M) (fun _=>dot) (parameters c) (grid (parameters c)) work before=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d (hd : d<96),after.mem (tag .output) (offset (c .output).strides b h n d)=
          .mk .bf16 (answer M dot (parameters c)
            (start (grid (parameters c)) (n/128,b*(parameters c).heads+h,0) before) (lane n d hd))) ∧
      RawFrame c before after := by
  have sep : (parameters c).Out≠(parameters c).Q ∧ (parameters c).Out≠(parameters c).K ∧
      (parameters c).Out≠(parameters c).V ∧ (parameters c).Out≠(parameters c).QS ∧
      (parameters c).Out≠(parameters c).KS := by simp [parameters,tag]
  let storage:=Machine.storage (parameters c) (domain c hc laws).canonical sep
  have unique : Unique (parameters c) (grid (parameters c)) work before := by
    intro a ha b hb i j hi hj same
    exact grid_unique _ storage before a (order.mem_iff.mpr ha) b (order.mem_iff.mpr hb) i j hi hj same
  obtain ⟨after,run,values,frame⟩:=run_correct (fun _=>M) (fun _=>dot) (parameters c)
    (Whole.separated _ storage) (grid (parameters c)) work before unique
  refine ⟨after,run,?_,?_⟩
  · intro b hb h hh n hn d hd
    have active : Source.active (parameters c)
        (start (grid (parameters c)) (n/128,b*(parameters c).heads+h,0) before) (lane n d hd) := by
      change n/128*128+n%128<(parameters c).length ∧ d<96
      exact ⟨by omega,hd⟩
    have covers:=(grid_covers (parameters c) b h n d hb hh hn hd).1
    have value:=values _ (order.mem_iff.mp covers) _ active
    rw [physical_address _ before b h n d hh hd] at value
    exact value
  · intro r off outside
    exact frame r off (exterior_outside c before work order r off outside)

def Stored (c : Call) (address : Nat) : Prop :=
  ∃ b h n d,Valid (parameters c) .output b h n d ∧ address=sourceAddress c .output b h n d
open Classical in
def observe {α β : Type} [Inhabited α] (decode : α→β) (c : Call)
    (before : Nat→β) (after : State α) (address : Nat) : β :=
  if Stored c address then decode ((after.mem (tag .output) ((address-base c .output)/2)).read .bf16)
  else before address
theorem observe_store {α β : Type} [Inhabited α] (decode : α→β) (c : Call)
    (before : Nat→β) (after : State α) (b h n d : Nat)
    (valid : Valid (parameters c) .output b h n d) :
    observe decode c before after (sourceAddress c .output b h n d)=
      decode ((after.mem (tag .output) (offset (c .output).strides b h n d)).read .bf16) := by
  have stored : Stored c (sourceAddress c .output b h n d):=⟨b,h,n,d,valid,rfl⟩
  rw [observe,if_pos stored]
  simp [sourceAddress,byteWidth,wordOffset]
theorem observe_frame {α β : Type} [Inhabited α] (decode : α→β) (c : Call)
    (before : Nat→β) (after : State α) (address : Nat) (outside : ¬Stored c address) :
    observe decode c before after address=before address := by simp [observe,outside]

theorem backing_exterior (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (hr : r∈reads) (address : Nat)
    (lo : (c r).storageBase≤address) (hi : address<(c r).storageBase+4*(c r).storageElements) :
    ¬Stored c address := by
  rintro ⟨b,h,n,d,valid,same⟩
  have apart:=laws.fresh r hr
  have bound:=laws.inBounds .output b h n d valid
  rw [←actual_address c hc laws .output b h n d valid] at same
  simp only [logicalAddress,byteWidth,ite_true] at same
  omega
theorem read_exterior (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (hr : r∈reads) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    ¬Stored c (logicalAddress c r b h n d) := by
  have width : byteWidth r=4 := by cases r <;> simp_all [reads,byteWidth]
  apply backing_exterior c hc laws r hr
  · simp [logicalAddress]
  · have bound:=laws.inBounds r b h n d valid
    simp only [logicalAddress,width]
    omega

theorem no_grid (c : Call) (empty : ¬Live (parameters c)) : programs (grid (parameters c))=[] := by
  by_cases hb : (parameters c).batches=0
  · simp [programs,grid,hb]
  by_cases hh : (parameters c).heads=0
  · simp [programs,grid,hh]
  have hn : (parameters c).length=0 := by unfold Live at empty;omega
  simp [programs,grid,hn]
theorem empty_observe {α β : Type} [Inhabited α] (decode : α→β) (c : Call)
    (empty : ¬Live (parameters c)) (before : Nat→β) (after : State α) (address : Nat) :
    observe decode c before after address=before address := by
  apply observe_frame
  rintro ⟨b,h,n,d,valid,_⟩
  have vn : n<(parameters c).length:=valid.2.2.1
  exact empty ⟨by have:=valid.1;omega,by have:=valid.2.1;omega,by omega⟩

#axiomsClean complete
#axiomsClean observe_store
#axiomsClean backing_exterior
#axiomsClean read_exterior
#axiomsClean no_grid
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly

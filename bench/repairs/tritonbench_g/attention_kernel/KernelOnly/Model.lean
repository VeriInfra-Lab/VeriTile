import bench.repairs.tritonbench_g.attention_kernel.Proofs.RawKernel
import bench.repairs.tritonbench_g.attention_kernel.KernelOnly.Inputs
import bench.repairs.tritonbench_g.attention_kernel.Proofs.Physical

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000

def roleOf (r : RegionName) : Option Role :=
  if r=tag .query then some .query else if r=tag .key then some .key
  else if r=tag .value then some .value else if r=tag .bias then some .bias
  else if r=tag .output then some .output else none

/-- All aliasing views consult one physical reader. No snapshot or copy is
introduced by these mathematical interpretations of original pointers. -/
def caller {α : Type} [Inhabited α] (c : Call) (read : Nat → α) (r : RegionName) (off : Nat) : α :=
  match roleOf r with | some role => read (base c role+2*off) | none => default
theorem caller_tag {α : Type} [Inhabited α] (c : Call) (read : Nat → α) (r : Role) (off : Nat) :
    caller c read (tag r) off=read (base c r+2*off) := by
  cases r <;> simp [caller,roleOf,tag]

def entry {α : Type} [Inhabited α] (c : Call) (read : RegionName → Nat → α) : State α where
  mem := fun r off => .mk (storage c).tile (ofFloat (storage c).float (read r off))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def RawFrame {α : Type} (c : Call) (before after : State α) : Prop :=
  ∀ r off,Physical.Exterior (parameters c) r off → after.mem r off=before.mem r off

theorem no_grid (c : Call) (hc : InputContract c) (empty : ¬Live (parameters c)) :
    programs (grid (parameters c))=[] := by
  have bm := Contract.positive_block _ _ _ hc
  by_cases hb : (parameters c).batches=0
  · simp [programs,grid,hb]
  by_cases hh : (parameters c).heads=0
  · simp [programs,grid,hh]
  have hn : (parameters c).length=0 := by unfold Live at empty;omega
  have divide : ((parameters c).length+(parameters c).bm-1)/(parameters c).bm=0 :=
    Nat.div_eq_of_lt (by omega)
  simp [programs,grid,divide]

/-- Success is derived for the full source and any complete-grid permutation.
The endpoint uses the kernel interpreter directly, without a host operation. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (c : Call) (hc : InputContract c) (before : State α) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) :
    ∃ after,runPrograms (fun _=>M) (fun _=>dot) (parameters c) (grid (parameters c)) work before=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d (hd : d<(parameters c).dim),
        after.mem (tag .output) (offset (c .output).strides b h n d)=
          .mk (storage c).tile (ofFloat (storage c).float
            (answer M dot (parameters c)
              (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) before)
              (lane (parameters c) (Contract.positive_block _ _ _ hc) n d hd)))) ∧
      RawFrame c before after := by
  exact RawKernel.complete M dot (parameters c) (Contract.positive_block _ _ _ hc)
    (Contract.domain _ _ _ hc).launch (by constructor <;> simp [parameters,tag]) before work order

def Stored (c : Call) (address : Nat) : Prop :=
  ∃ b h n d,Valid (parameters c) .output b h n d ∧ address=sourceAddress c .output b h n d
open Classical in
def observe {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (before : Nat → β) (after : State α) (address : Nat) : β :=
  if Stored c address then
    decode (toFloat (storage c).float
      ((after.mem (tag .output) ((address-base c .output)/2)).read (storage c).tile))
  else before address

theorem observe_store {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (before : Nat → β) (after : State α) (b h n d : Nat)
    (valid : Valid (parameters c) .output b h n d) :
    observe decode c before after (sourceAddress c .output b h n d)=
      decode (toFloat (storage c).float
        ((after.mem (tag .output) (offset (c .output).strides b h n d)).read (storage c).tile)) := by
  have stored : Stored c (sourceAddress c .output b h n d) := ⟨b,h,n,d,valid,rfl⟩
  rw [observe,if_pos stored]
  simp [sourceAddress]

theorem observe_frame {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (before : Nat → β) (after : State α) (address : Nat) (outside : ¬Stored c address) :
    observe decode c before after address=before address := by simp [observe,outside]

theorem read_exterior (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (hr : r∈reads) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    ¬Stored c (logicalAddress c r b h n d) := by
  rintro ⟨z,k,m,e,hout,same⟩
  have live : Live (parameters c) := ⟨by have:=valid.1;omega,
    by have:=valid.2.1;omega,by have:=valid.2.2.1;omega⟩
  have separated := (hc.2 live).2
  have member : span c r∈(arguments c).readSpans := List.mem_map.mpr ⟨r,hr,rfl⟩
  have apart := separated (span c r) member
  have ib := laws.inBounds r b h n d valid
  have ob := laws.inBounds .output z k m e hout
  rw [←actual_address c hc laws .output z k m e hout] at same
  simp only [Contract.Separate] at separated
  simp only [arguments,span,logicalAddress] at apart same
  omega

theorem empty_observe {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (empty : ¬Live (parameters c)) (before : Nat → β) (after : State α) (address : Nat) :
    observe decode c before after address=before address := by
  apply observe_frame
  rintro ⟨b,h,n,d,valid,_⟩
  exact empty ⟨by have:=valid.1;omega,by have:=valid.2.1;omega,by have:=valid.2.2.1;omega⟩

/-- Checked separation preserves each complete read allocation, including
prefixes, gaps and trailing storage, not just its logical Tensor elements. -/
theorem backing_exterior (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (hr : r∈reads) (address : Nat)
    (lo : (c r).storageBase≤address) (hi : address<(c r).storageBase+2*(c r).storageElements) :
    ¬Stored c address := by
  rintro ⟨b,h,n,d,valid,same⟩
  have live : Live (parameters c) := ⟨by have:=valid.1;omega,
    by have:=valid.2.1;omega,by have:=valid.2.2.1;omega⟩
  have member : span c r∈(arguments c).readSpans := List.mem_map.mpr ⟨r,hr,rfl⟩
  have apart := (hc.2 live).2 (span c r) member
  have bound := laws.inBounds .output b h n d valid
  rw [←actual_address c hc laws .output b h n d valid] at same
  simp only [arguments,span,logicalAddress] at apart same
  omega

#axiomsClean complete
#axiomsClean observe_store
#axiomsClean observe_frame
#axiomsClean read_exterior
#axiomsClean empty_observe
#axiomsClean backing_exterior
end
end VeriTile.Bench.Repairs.AttentionKernel.KernelOnly

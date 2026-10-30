import bench.tritonbench_g.attention_kernel_aligned.Proofs.Inputs
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly

open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue

open VeriTile.Bench.Repairs.AttentionKernelAligned Source Whole
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

def roleOf (r : RegionName) : Option Role :=
  if r=tag .query then some .query else if r=tag .key then some .key
  else if r=tag .value then some .value else if r=tag .bias then some .bias
  else if r=tag .output then some .output else none
/-- One physical reader at the original pointers; no input copies. -/
def caller {α : Type} [Inhabited α] (c : Call) (read : Nat → α) (r : RegionName) (off : Nat) : α :=
  match roleOf r with | some role=>read (base c role+2*off) | none=>default
theorem caller_tag {α : Type} [Inhabited α] (c : Call) (read : Nat → α) (r : Role) (off : Nat) :
    caller c read (tag r) off=read (base c r+2*off) := by
  cases r <;> simp [caller,roleOf,tag]
def entry {α : Type} [Inhabited α] (read : RegionName → Nat → α) : State α where
  mem:=fun r off=>.mk .fp16 (read r off)
  regs:=fun _ _ _=>none
  pids:=fun _=>0
  numPids:=fun _=>1
  undef:=fun d _ _=>defaultValue d

def RawFrame {α : Type} (c : Call) (before after : State α) : Prop :=
  ∀ r off,OutsideLogical (parameters c) r off → after.mem r off=before.mem r off

theorem no_grid (c : Call) (hc : InputContract c) (empty : ¬Whole.Nonempty (parameters c)) :
    programs (grid (parameters c))=[] := by
  have bm:=(guard c hc).selected.blockM_pos (parameters c)
  by_cases hb : (parameters c).batches=0
  · simp [programs,grid,hb]
  by_cases hh : (parameters c).heads=0
  · simp [programs,grid,hh]
  have hn : (parameters c).length=0 := by unfold Whole.Nonempty at empty;omega
  have divide : ((parameters c).length+(parameters c).blockM-1)/(parameters c).blockM=0 :=
    Nat.div_eq_of_lt (by omega)
  simp [programs,grid,divide]

/-- Direct execution success and every logical store for any full-grid order. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (c : Call) (hc : InputContract c) (before : State α) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) :
    ∃ after,runPrograms (fun _=>M) (fun _=>P) (parameters c) (grid (parameters c)) work before=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d (hd : d<(parameters c).head),
        after.mem (tag .output) (offset (c .output).strides b h n d)=.mk .fp16
          (answer P (parameters c) (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) before)
            (lane (parameters c) (guard c hc).selected n d hd))) ∧ RawFrame c before after := by
  by_cases live : Whole.Nonempty (parameters c)
  · have unique : Unique (parameters c) (grid (parameters c)) work before := by
      intro a ha b hb i j hi hj same
      exact grid_unique _ (guard c hc) before a (order.mem_iff.mpr ha) b (order.mem_iff.mpr hb) i j hi hj same
    have separate : Separated (parameters c) := by constructor <;> simp [parameters,tag]
    obtain ⟨after,run,values,frame⟩:=run_correct (fun _=>M) (fun _=>P) (parameters c) separate
      ((guard c hc).selected.blockN_pos _) (grid (parameters c)) work before unique
    refine ⟨after,run,?_,?_⟩
    · intro b hb h hh n hn d hd
      have active : Whole.active (parameters c)
          (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) before)
          (lane (parameters c) (guard c hc).selected n d hd) := by
        change n/c.bm*c.bm+n%c.bm<(parameters c).length
        simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
      have covers:=(grid_covers (parameters c) (guard c hc).selected b h n hb hh hn).1
      have value:=values _ (order.mem_iff.mp covers) _ active
      rw [physical_address _ (guard c hc).selected before b h n d hh hd] at value
      exact value
    · intro r off outside
      apply frame
      rcases outside with outside|outside
      · exact Or.inl outside
      · right
        intro a ha i hi he
        obtain ⟨hx,hy,hz⟩:=(mem_programs (grid (parameters c)) a.1 a.2.1 a.2.2).mp (order.mem_iff.mpr ha)
        obtain ⟨_,hB,hH⟩:=head_decode (parameters c) a.2.1 hy
        exact outside _ hB _ hH _ hi _ i.2.1.isLt he
  · have nil : work=[] := List.nil_perm.mp (by simpa [no_grid c hc live] using order)
    refine ⟨before,by simp [nil,runPrograms],?_,fun _ _ _=>rfl⟩
    intro b hb h hh n hn d hd
    exact False.elim (live ⟨by omega,by omega,by omega⟩)

def Stored (c : Call) (address : Nat) : Prop :=
  ∃ b h n d,Valid (parameters c) .output b h n d ∧ address=sourceAddress c .output b h n d
open Classical in
def observe {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (before : Nat → β) (after : State α) (address : Nat) : β :=
  if Stored c address then decode ((after.mem (tag .output) ((address-base c .output)/2)).read .fp16)
  else before address
theorem observe_store {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (before : Nat → β) (after : State α) (b h n d : Nat) (valid : Valid (parameters c) .output b h n d) :
    observe decode c before after (sourceAddress c .output b h n d)=
      decode ((after.mem (tag .output) (offset (c .output).strides b h n d)).read .fp16) := by
  rw [observe,if_pos (show Stored c (sourceAddress c .output b h n d) from ⟨b,h,n,d,valid,rfl⟩)]
  simp [sourceAddress]
theorem observe_frame {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (before : Nat → β) (after : State α) (address : Nat) (outside : ¬Stored c address) :
    observe decode c before after address=before address := by simp [observe,outside]

theorem backing_exterior (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (hr : r∈reads) (address : Nat)
    (lo : (c r).storageBase≤address) (hi : address<(c r).storageBase+2*(c r).storageElements) :
    ¬Stored c address := by
  rintro ⟨b,h,n,d,valid,same⟩
  have live : Whole.Nonempty (parameters c) := ⟨by have:=valid.1;omega,
    by have:=valid.2.1;omega,by have:=valid.2.2.1;omega⟩
  have ne : r≠.output := by cases r <;> simp_all [reads]
  have apart:=(hc.checked.2.2.2 live).2.2.2.2.2.2.2.2 r .output ne
  have bound:=laws.inBounds .output b h n d valid
  rw [←actual_address c hc laws .output b h n d valid] at same
  simp only [observations,logicalAddress] at apart same
  omega
theorem read_exterior (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (hr : r∈reads) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    ¬Stored c (logicalAddress c r b h n d) := by
  apply backing_exterior c hc laws r hr
  · simp [logicalAddress]
  · have bound:=laws.inBounds r b h n d valid
    simp only [logicalAddress]
    omega
theorem empty_observe {α β : Type} [Inhabited α] (decode : α → β) (c : Call)
    (empty : ¬Whole.Nonempty (parameters c)) (before : Nat → β) (after : State α) (address : Nat) :
    observe decode c before after address=before address := by
  apply observe_frame
  rintro ⟨b,h,n,d,valid,_⟩
  exact empty ⟨by have:=valid.1;omega,by have:=valid.2.1;omega,by have:=valid.2.2.1;omega⟩

#axiomsClean complete
#axiomsClean backing_exterior
#axiomsClean read_exterior
#axiomsClean observe_store
#axiomsClean observe_frame
#axiomsClean empty_observe
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly

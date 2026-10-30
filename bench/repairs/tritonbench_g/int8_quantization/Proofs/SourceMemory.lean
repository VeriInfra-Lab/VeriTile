import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceObservation
import bench.repairs.tritonbench_g.int8_quantization.Proofs.Words

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Bytes
open VeriTile.Triton FP.Structural
open Int8Quantization.Words
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def read32 (m : Memory) (a : Nat) : Word := fun b=> m (a+b.val)

structure Layout where
  inputBase : Nat
  outputBase : Nat
  scaleBase : Nat

def outputWindow (p : Parameters) (l : Layout) (a : Nat) : Prop :=
  l.outputBase≤ a ∧ a< l.outputBase+p.batches*p.length*p.channels
def scaleWindow (p : Parameters) (l : Layout) (a : Nat) : Prop :=
  l.scaleBase≤ a ∧ a< l.scaleBase+4*(p.batches*p.blocks)
instance (p : Parameters) (l : Layout) (a : Nat) : Decidable (outputWindow p l a) :=
  inferInstanceAs (Decidable (_ ∧ _))
instance (p : Parameters) (l : Layout) (a : Nat) : Decidable (scaleWindow p l a) :=
  inferInstanceAs (Decidable (_ ∧ _))

structure Bounds (p : Parameters) (l : Layout) : Prop where
  input : l.inputBase+4*(p.batches*p.length*p.channels)≤2^64
  output : l.outputBase+p.batches*p.length*p.channels≤2^64
  scale : l.scaleBase+4*(p.batches*p.blocks)≤2^64
  separated : ∀ a,outputWindow p l a→¬scaleWindow p l a

/-- Standard typed ABI decoding of the three actual contiguous allocations.
There is no output formula in this initial state. -/
def bind (l : Layout) (m : Memory) : State Word where
  mem:=fun r o=> if r=xRegion then .mk .real (read32 m (l.inputBase+4*o))
    else if r=outRegion then .mk .int ((m (l.outputBase+o)).toInt)
    else if r=scaleRegion then .mk .real (read32 m (l.scaleBase+4*o))
    else .mk .real (fun _=>0)
  regs:=fun _ _ _=> none
  pids:=fun _=>0
  numPids:=fun _=>0
  undef:=fun d _ _=> defaultValue d

/-- Ordinary store-channel encoding. Re-encoding the unchanged bound state
is the identity, proved below; only the actual source's changed typed cells
can change bytes. The output uses one byte and each scale uses four bytes. -/
def store (p : Parameters) (l : Layout) (before : Memory) (typed : State Word) : Memory :=
  fun a=> if outputWindow p l a then
      BitVec.ofInt 8 ((typed.mem outRegion (a-l.outputBase)).read .int)
    else if scaleWindow p l a then
      ((typed.mem scaleRegion ((a-l.scaleBase)/4)).read .real) ⟨(a-l.scaleBase)%4,by omega⟩
    else before a

theorem store_identity (p : Parameters) (l : Layout) (before : Memory) :
    store p l before (bind l before)=before := by
  funext a
  unfold store
  split_ifs with ho hs
  · have lower:=ho.1
    simp [bind,show outRegion≠xRegion by decide,Cell.read,show TileDType.int≠.real by decide]
    rw [Nat.add_sub_of_le lower]
  · have lower:=hs.1
    simp [bind,show scaleRegion≠xRegion by decide,show scaleRegion≠outRegion by decide,read32]
    congr 1
    have eq:=Nat.mod_add_div (a-l.scaleBase) 4
    omega
  · rfl

theorem store_output (p : Parameters) (l : Layout) (before : Memory) (typed : State Word)
    (o : Nat) (bound : o< p.batches*p.length*p.channels) :
    store p l before typed (l.outputBase+o)=BitVec.ofInt 8 ((typed.mem outRegion o).read .int) := by
  simp [store,outputWindow,bound]

theorem store_scale (p : Parameters) (l : Layout) (bounds : Bounds p l)
    (before : Memory) (typed : State Word) (g : Nat) (bound : g< p.batches*p.blocks) :
    read32 (store p l before typed) (l.scaleBase+4*g)=(typed.mem scaleRegion g).read .real := by
  funext b
  have scale : scaleWindow p l (l.scaleBase+4*g+b.val) := by
    have h:=b.isLt;unfold scaleWindow;omega
  have output : ¬outputWindow p l (l.scaleBase+4*g+b.val) := fun h=> bounds.separated _ h scale
  simp only [read32,store,if_neg output,if_pos scale]
  have off : l.scaleBase+4*g+b.val-l.scaleBase=4*g+b.val := by omega
  have div : (4*g+b.val)/4=g := by omega
  have mod : (4*g+b.val)%4=b.val := by omega
  simp only [off,div]
  apply congrArg ((typed.mem scaleRegion g).read .real)
  apply Fin.ext
  exact mod

theorem store_frame (p : Parameters) (l : Layout) (before : Memory) (typed : State Word)
    (a : Nat) (outsideOutput : ¬outputWindow p l a) (outsideScale : ¬scaleWindow p l a) :
    store p l before typed a=before a := by simp [store,outsideOutput,outsideScale]

/-- Full literal native access, including widening and byte scaling. -/
def nativeAccess (p : Parameters) (l : Layout) (pid lane : Nat) : Option (Nat×Nat) :=
  let n:=Native.nativePrefix p pid lane
  if n.lane<(p.blockRows*p.channels:Int) ∧ n.row<(p.length:Int) then
    some (Native.pointer 4 l.inputBase n.offset,Native.pointer 1 l.outputBase n.offset)
  else none

def exactAccess (p : Parameters) (l : Layout) (pid lane : Nat) : Option (Nat×Nat) :=
  let g:=pid+p.base
  let o:=g/p.blocks*p.length*p.channels+g%p.blocks*p.blockRows*p.channels+lane
  if lane< p.blockRows*p.channels ∧ g%p.blocks*p.blockRows+lane/p.channels< p.length then
    some (l.inputBase+4*o,l.outputBase+o)
  else none

theorem access_exact (p : Parameters) (l : Layout) (a : Native.Admitted p) (bounds : Bounds p l)
    (pid lane : Nat) (pidLimit : pid<2^31)
    (group : pid+p.base< p.batches*p.blocks) (laneBound : lane< p.width) :
    nativeAccess p l pid lane=exactAccess p l pid lane := by
  unfold nativeAccess
  rw [Native.prefix_exact p a pid lane pidLimit group laneBound]
  simp only [Native.exactPrefix]
  norm_cast
  by_cases active : lane< p.blockRows*p.channels ∧
      (pid+p.base)%p.blocks*p.blockRows+lane/p.channels< p.length
  · have blocks : 0< p.blocks := by nlinarith
    have batch : (pid+p.base)/p.blocks< p.batches := (Nat.div_lt_iff_lt_mul blocks).mpr group
    have offset:=Geometry.active_offset p.batches p.length p.channels p.blockRows
      ((pid+p.base)/p.blocks) ((pid+p.base)%p.blocks) lane a.channelsPositive batch active.2
    simp only [active,↓reduceIte,exactAccess,and_self]
    rw [Native.pointer_exact 4 _ _ _ (by decide) bounds.input offset,
      Native.pointer_exact 1 _ _ _ (by decide) (by simpa using bounds.output) offset]
    simp
  · simp [exactAccess,active]

theorem scale_pointer_exact (p : Parameters) (l : Layout) (bounds : Bounds p l)
    (g : Nat) (active : g< p.batches*p.blocks) :
    Native.pointer 4 l.scaleBase g=l.scaleBase+4*g :=
  Native.pointer_exact 4 _ _ _ (by decide) bounds.scale active

#axiomsClean store_identity
#axiomsClean store_output
#axiomsClean store_scale
#axiomsClean store_frame
#axiomsClean access_exact
#axiomsClean scale_pointer_exact
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Bytes

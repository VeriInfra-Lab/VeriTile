import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedIntegerMacro
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedDimensionBinding

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.BoundSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Memory
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

/-- Selected-request scalar laws, never an execution or output premise. In
particular the FP64 codec excludes reversing the order of FP32-origin values. -/
inductive Binding (casts : Request.Casts) : Request.Kind → Type
  | floating (kind : RealScratch.FloatingKind) (round : TypedScratch.Rounding)
      (codec : TypedScalarBinding.FloatCodec kind round (casts.floating kind)) :
      Binding casts (.floating kind)
  | integer (kind : Formats.IntegerKind)
      (zero : ConversionMacro.ZeroClamp kind (casts.integer kind)) : Binding casts (.integer kind)

/-- The basic FP64 codec already enforces scalar order preservation for all
FP32-origin values. A cast that swaps 1 and 2 cannot inhabit this interface. -/
theorem fp64_cast_order (casts : Request.Casts) (round : TypedScratch.Rounding)
    (codec : TypedScalarBinding.FloatCodec .fp64 round (casts.floating .fp64)) (x y : Word) :
    Reduction.orderedGt (floatKey .fp64 (casts.floating .fp64 x))
      (floatKey .fp64 (casts.floating .fp64 y))=greater x y := by
  have relation := codec.greater (casts.floating .fp64 x) (casts.floating .fp64 y)
    (Or.inl ⟨x,rfl⟩) (Or.inl ⟨y,rfl⟩)
  rw [codec.castValue,codec.castValue] at relation
  exact relation.symm

def X : RegionName := ⟨"argmax.input"⟩
def V : RegionName := ⟨"argmax.scratch"⟩
def VI : Region .int := ⟨"argmax.scratch"⟩
def I : Region .int := ⟨"argmax.indices"⟩
def O : Region .int := ⟨"argmax.output"⟩

def Binding.cell {casts kind} : Binding casts kind → MidWord kind.format → Cell Word
  | .floating k _ codec => fun word => .mk (TypedScratch.channel k).toTileDType
      (ofFloat (TypedScratch.channel k) (codec.decode word))
  | .integer k _ => fun word => .mk .int (Formats.decodeStored k word)

def Binding.ops {casts kind} : Binding casts kind → Algebra Word → Algebra Word
  | .floating _ round _ => TypedScratch.backend round
  | .integer k _ => TypedIntegerMacro.backend k (casts.integer k)

def Binding.first {casts kind} : Binding casts kind → GlobalConfig → Bool → ComputeKernel
  | .floating k _ _,c,index64 => TypedScratch.firstKernel k X V I c.size c.block 0 index64
  | .integer _ _,c,index64 => TypedIntegerMacro.kernel X V I c.size c.block 0 index64

def Binding.second {casts kind} : Binding casts kind → GlobalConfig → ComputeKernel
  | .floating k _ _,c => TypedScratch.secondKernel k V I O c.blocks c.midBlock
  | .integer k _,c => TypedInteger.stage2 VI I O c.blocks c.midBlock
      (Formats.decodeStored k (integerScratch k (casts.integer k)).padding)

/-- All entry loads come from this exact current byte state. Other registers
and undefined channels are arbitrary block-local template values. -/
def entry {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (p : GlobalMemory.Buffers) (before : M) (template : State Word) (pid : Nat) : State Word :=
  {template with
    pids:=fun ax => if ax=0 then pid else template.pids ax
    mem:=fun region offset =>
      if region=X then .mk .real (mem.read 4 before (address p.input 4 offset))
      else if region=V then binding.cell (mem.read kind.format.bytes before (address p.midValue kind.format.bytes offset))
      else if region=Region.cast I then .mk .int (Int.ofNat (mem.read 8 before (address p.midIndex 8 offset)).toNat)
      else if region=Region.cast O then .mk .int (Int.ofNat (mem.read 8 before (address p.output 8 offset)).toNat)
      else template.mem region offset}

@[simp] theorem entry_input {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (p : GlobalMemory.Buffers) (before : M) (template : State Word) (pid offset : Nat) :
    ((entry mem binding p before template pid).mem X offset).read .real=GlobalMemory.input mem p before offset := by
  simp [entry,GlobalMemory.input]

@[simp] theorem entry_pid {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (p : GlobalMemory.Buffers) (before : M) (template : State Word) (pid : Nat) :
    (entry mem binding p before template pid).pids 0=pid := by simp [entry]

@[simp] theorem entry_mid {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (p : GlobalMemory.Buffers) (before : M) (template : State Word) (pid offset : Nat) :
    (entry mem binding p before template pid).mem V offset=
      binding.cell (mem.read kind.format.bytes before (address p.midValue kind.format.bytes offset)) := by
  simp [entry,X,V]

@[simp] theorem entry_index {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (p : GlobalMemory.Buffers) (before : M) (template : State Word) (pid offset : Nat) :
    ((entry mem binding p before template pid).mem (Region.cast I) offset).read .int=
      Int.ofNat (mem.read 8 before (address p.midIndex 8 offset)).toNat := by
  simp [entry,X,V,I,Region.cast]

/-- A complete quotation followed by exactly its two scalar physical stores.
The raw scratch scalar is the requested cast of the original-address reread;
the quotation stores its codec representation and the same selected index. -/
def FirstStep {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (rest : Algebra Word) (template : State Word) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (index64 : Bool) (pid : Nat) (before after : M) : Prop :=
  ∃ quoted index, FP.Structural.exec (binding.ops rest) (binding.first c index64)
      (entry mem binding p before template pid)=some quoted ∧
    quoted.mem=(((entry mem binding p before template pid).write V pid
      (binding.cell ((Request.scratch casts kind).cast (GlobalMemory.input mem p before index)))).write
        (Region.cast I) pid (.mk .int (Int.ofNat index))).mem ∧
    after=mem.write 8 (mem.write kind.format.bytes before (address p.midValue kind.format.bytes pid)
      ((Request.scratch casts kind).cast (GlobalMemory.input mem p before index)))
      (address p.midIndex 8 pid) (BitVec.ofNat 64 index)

theorem first_step {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (rest : Algebra Word) (template : State Word) (c : GlobalConfig) (g : Native.GlobalPrograms c)
    (p : GlobalMemory.Buffers) (positive : 0<c.block) (index64 : Bool) (pid : Nat) (before : M) :
    FirstStep mem binding rest template c p index64 pid before
      (Native.firstProgram mem (Request.scratch casts kind) g p before pid) := by
  cases binding with
  | floating k round codec =>
    obtain ⟨t,run,writes⟩ := TypedNative.first_float k round (casts.floating k) codec rest
      (entry mem (.floating k round codec) p before template pid) X V I c g 0 index64 positive
    simp only [entry_input,entry_pid,Nat.add_zero] at writes
    refine ⟨t,Native.firstIndex g (GlobalMemory.input mem p before) pid,run,?_,rfl⟩
    exact writes
  | integer k zero =>
    obtain ⟨t,run,writes⟩ := TypedIntegerMacro.source_native k (casts.integer k) rest
      (entry mem (.integer k zero) p before template pid) X V I c g 0 index64 positive
    simp only [entry_input,entry_pid,Nat.add_zero] at writes
    refine ⟨t,Native.firstIndex g (GlobalMemory.input mem p before) pid,run,?_,rfl⟩
    exact writes

def Reachable {casts kind} (binding : Binding casts kind) (word : MidWord kind.format) : Prop :=
  match binding with
  | .floating k _ _ => TypedScalarBinding.Reachable k (casts.floating k) word
  | .integer _ _ => True

/-- The second quotation reads current scratch/index bytes and commits the
exact gathered output scalar to that same memory. Its complete typed frame is
the equality to the single displayed store. -/
def SecondStep {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (rest : Algebra Word) (template : State Word) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (before after : M) : Prop :=
  ∃ quoted, ∃ payload : BitVec 64, FP.Structural.exec (binding.ops rest) (binding.second c)
      (entry mem binding p before template 0)=some quoted ∧
    quoted.mem=((entry mem binding p before template 0).write (Region.cast O) 0
      (.mk .int (Int.ofNat payload.toNat))).mem ∧
    after=mem.write 8 before (address p.output 8 0) payload

theorem second_step {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (rest : Algebra Word) (template : State Word) (c : GlobalConfig) (g : Native.GlobalPrograms c)
    (p : GlobalMemory.Buffers) (positive : 0<c.midBlock) (before : M)
    (reachable : ∀ i,i<c.blocks → Reachable binding (mem.read kind.format.bytes before (address p.midValue kind.format.bytes i))) :
    SecondStep mem binding rest template c p before
      (Native.secondProgram mem (Request.scratch casts kind) kind.isFloating g p before) := by
  cases binding with
  | floating k round codec =>
    obtain ⟨t,run,writes⟩ := TypedNative.second_float k round (casts.floating k) codec rest
      (entry mem (.floating k round codec) p before template 0) V I O c g positive
      (fun i => mem.read k.format.bytes before (address p.midValue k.format.bytes i)) reachable
      (by intro i hi; rw [entry_mid]; cases k <;> rfl)
    simp only [entry_index] at writes
    exact ⟨t,_,run,writes,rfl⟩
  | integer k zero =>
    obtain ⟨t,run,writes⟩ := TypedNative.second_integer k (casts.integer k)
      (Binding.ops (.integer k zero) rest) (entry mem (.integer k zero) p before template 0) VI I O c g positive
      (fun i => mem.read (Formats.format k).bytes before (address p.midValue (Formats.format k).bytes i))
      (by intro i hi; change ((entry mem (.integer k zero) p before template 0).mem V i).read .int=_
          rw [entry_mid]; rfl)
    simp only [entry_index] at writes
    exact ⟨t,_,run,writes,rfl⟩

#axiomsClean first_step
#axiomsClean second_step
#axiomsClean fp64_cast_order
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.BoundSource

import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.QKV
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.PhysicalMemory
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedMemory

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

/-- Q is supplied as an ordinary existing destination. Its offset may be
nonzero; no allocation success or fresh-allocation frame is assumed. -/
structure ExistingQ (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) : Prop where
  ordinary : RmsMatmulRbe.Checked.Ordinary q
  admitted : RmsMatmulRbe.Contract.TensorAdmitted q [RmsMatmulRbe.Contract.fp16]
  shape : q.metadata.shape=[(RmsMatmulRbe.Checked.shape i.original).B,(RmsMatmulRbe.Checked.shape i.original).M,
    (RmsMatmulRbe.Checked.shape i.original).N]
  device : q.metadata.device=i.original.x.metadata.device
  separate : ∀ t∈i.original.reads++[i.original.k,i.original.v],RmsMatmulRbe.Contract.disjoint q t

def OrdinaryInputs (i : Contract.Inputs) : Prop :=
  ∀ t∈i.original.reads++[i.original.k,i.original.v],RmsMatmulRbe.Checked.Ordinary t

def weightRegion (phase : Fin 3) : RegionName :=
  if phase.val=0 then "QW" else if phase.val=1 then "KW" else "VW"
def outputRegion (phase : Fin 3) : RegionName :=
  if phase.val=0 then "Q" else if phase.val=1 then "K" else "V"

def useFP8 (i : Contract.Inputs) : Bool := decide (i.original.qw.metadata.dtype=RmsMatmulRbe.Contract.int8)

def arguments (i : Contract.Inputs) (phase : Fin 3) : Arguments :=
  let s := RmsMatmulRbe.Checked.shape i.original
  { X:="X",W:=weightRegion phase,RMS:="RMS",OUT:=outputRegion phase,
    M:=s.M,N:=s.N,K:=s.K,sxb:=s.M*s.K,sxm:=s.K,sxk:=1,
    swk:=1,swn:=s.K,srms:=1,sob:=s.M*s.N,som:=s.N,son:=1,
    BM:=16,BN:=64,BK:=64,headDim:=s.headDim,start:=i.original.scalars.start,
    rotary:=decide (phase.val<2) }

theorem arguments_apart (i : Contract.Inputs) (writer reader : Fin 3) :
    (arguments i reader).X≠(arguments i writer).OUT ∧
    (arguments i reader).W≠(arguments i writer).OUT ∧
    (arguments i reader).RMS≠(arguments i writer).OUT := by
  fin_cases writer <;> fin_cases reader <;> simp [arguments,weightRegion,outputRegion]

theorem arguments_even (i : Contract.Inputs) (h : Contract.InputContract i) (phase : Fin 3) :
    (arguments i phase).N%2=0 := by
  have f := RmsMatmulRbe.Checked.facts i.original h.1
  rcases f.valid with ⟨hK,hh,hd,he,hN⟩
  change (RmsMatmulRbe.Checked.shape i.original).N%2=0
  rw [hN,Nat.mul_mod,he,Nat.mul_zero,Nat.zero_mod]

def address (t : RmsMatmulRbe.Contract.Tensor) (element : Nat) : Nat :=
  t.base+RmsMatmulRbe.Contract.width t.metadata.dtype*(t.offset+element)

def inBacking (t : RmsMatmulRbe.Contract.Tensor) (element : Nat) : Prop :=
  RmsMatmulRbe.Contract.width t.metadata.dtype*(t.offset+element+1)≤t.bytes
instance (t : RmsMatmulRbe.Contract.Tensor) (element : Nat) : Decidable (inBacking t element) :=
  inferInstanceAs (Decidable (_≤_))

def sourceByte (access : Access α Store Byte) (t : RmsMatmulRbe.Contract.Tensor)
    (s : Store) (element : Nat) : Int :=
  if inBacking t element then (access.read8 s (address t element)).toInt else 0

structure Program where
  batch : Nat
  block : Nat
  deriving DecidableEq

def Program.pids (id : Program) : Nat → Nat
  | 0 => id.batch
  | 1 => id.block
  | _ => 0

def Program.Valid (i : Contract.Inputs) (id : Program) : Prop :=
  id.batch<(RmsMatmulRbe.Checked.shape i.original).B ∧
  id.block<RmsMatmulRbe.tilesM (RmsMatmulRbe.Checked.shape i.original)*RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original)

def programs (i : Contract.Inputs) : List Program :=
  (List.range (RmsMatmulRbe.Checked.shape i.original).B).flatMap (fun b =>
    (List.range (RmsMatmulRbe.tilesM (RmsMatmulRbe.Checked.shape i.original)*
      RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original))).map (fun block => ⟨b,block⟩))

theorem programs_mem (i : Contract.Inputs) (id : Program) : id∈programs i ↔ id.Valid i := by
  cases id
  simp [programs,Program.Valid]

def weightCell (access : Access α Store Byte) (zero : α) (fp8 : Bool) (t : RmsMatmulRbe.Contract.Tensor)
    (s : Store) (element : Nat) : Cell α :=
  if fp8 then .mk .int (sourceByte access t s element)
  else .mk .fp16 (if inBacking t element then access.read16 s (address t element) else zero)

def destinationCell (access : Access α Store Byte) (zero : α) (fp8 : Bool) (t : RmsMatmulRbe.Contract.Tensor)
    (s : Store) (element : Nat) : Cell α :=
  if fp8 then .mk .f8e5 (if inBacking t element then access.readE5 s (address t element) else zero)
  else .mk .fp16 (if inBacking t element then access.read16 s (address t element) else zero)

/-- Every logical source region observes the same caller physical store.
Read aliases therefore see the same bytes and storage offsets. -/
def state {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) : State α where
  mem := fun region element =>
    if region="X" then .mk .real (if inBacking i.original.x element then
      access.read32 s (address i.original.x element) else zero)
    else if region="QW" then weightCell access zero (useFP8 i) i.original.qw s element
    else if region="KW" then weightCell access zero (useFP8 i) i.original.kw s element
    else if region="VW" then weightCell access zero (useFP8 i) i.original.vw s element
    else if region="RMS" then weightCell access zero (useFP8 i) i.original.rms s element
    else if region="Q" then destinationCell access zero Bool.false q s element
    else if region="K" then destinationCell access zero (useFP8 i) i.original.k s element
    else if region="V" then destinationCell access zero (useFP8 i) i.original.v s element
    else .mk .real zero
  regs := fun _ _ _ => none
  pids := id.pids
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem byte_int8 (value : BitVec 8) : Float8E5Bits.isInt8 value.toInt=Bool.true := by
  have lo := BitVec.le_toInt (x:=value)
  have hi := BitVec.toInt_lt (x:=value)
  simp only [Float8E5Bits.isInt8,decide_eq_true_eq]
  exact ⟨lo,hi⟩

theorem sourceByte_int8 (access : Access α Store Byte) (t : RmsMatmulRbe.Contract.Tensor)
    (s : Store) (element : Nat) : Float8E5Bits.isInt8 (sourceByte access t s element)=Bool.true := by
  unfold sourceByte
  split_ifs
  · exact byte_int8 _
  · decide

theorem zero_int8 : Float8E5Bits.isInt8 0=Bool.true := by decide

theorem state_int8 {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program) :
    Int8Memory (state access zero i q s id) := by
  intro r o
  simp only [state]
  split_ifs
  all_goals cases hf : useFP8 i <;>
    simp [weightCell,destinationCell,hf,Cell.read,defaultValue,sourceByte_int8,zero_int8]

#axiomsClean arguments_apart
#axiomsClean arguments_even
#axiomsClean programs_mem
#axiomsClean state_int8
#specNonCircular state avoiding [qkvCode,qkvWrite,execution,calleeTarget]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel

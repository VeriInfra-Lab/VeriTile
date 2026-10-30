import bench.tritonbench_g.attn_fwd_causal.Proofs.NativeProjection
namespace VeriTile.Bench.Repairs.AttnFwdCausal.NativeReference
open VeriTile Triton FP.Structural Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- Logical FP32 word readers for one batch/head, independent of registers,
source execution, source accumulators, allocation and final memory. -/
structure Inputs where
  (query key value : Nat → Nat → Word)
  (queryScale keyScale : Nat → Word)

structure Row where
  (maximum normalizer : Fin 4 → Graph)
  value : Fin 128 → Fin 4 → Graph

def initial : Row :=
  ⟨fun _=>.bits 0xff800000,fun _=>.bits 0x3f800000,fun _ _=>zero⟩

def score (length queryRow start : Nat) (input : Inputs) (j : Fin 64) : Graph :=
  if j.val<length-start ∧ start+j.val≤queryRow then
    .mul32 (.bits (input.keyScale (start/64)))
      (.mul32
        (dotTF32 128 (queryRow%128) j.val
          (fun d=>if d.val<128 then .bits (input.query queryRow d.val) else zero)
          (fun d=>if d.val<128 then .bits (input.key (start+j.val) d.val) else zero))
        (.bits (input.queryScale (queryRow/128))))
  else .bits 0xff800000

def maximum (length queryRow start : Nat) (input : Inputs) (old : Row) (owner : Fin 4) : Graph :=
  .maxNum32 (old.maximum owner) ((plan owner).value Graph.maxNum32 (score length queryRow start input))

def probability (length queryRow start : Nat) (input : Inputs) (old : Row)
    (owner : Fin 4) (j : Fin 64) : Graph :=
  .exp2ApproxFTZ (.sub32 (score length queryRow start input j)
    (maximum length queryRow start input old owner))

def alpha (length queryRow start : Nat) (input : Inputs) (old : Row) (owner : Fin 4) : Graph :=
  .exp2ApproxFTZ (.sub32 (old.maximum owner) (maximum length queryRow start input old owner))

def contribution (length queryRow start : Nat) (input : Inputs) (old : Row) (feature : Fin 128) : Graph :=
  .widen16 (dotHalf 64 (queryRow%128) feature.val
    (fun j=>.narrow16 (probability length queryRow start input old (physicalOwner j.val) j))
    (fun j=>.narrow16 (if j.val<length-start ∧ feature.val<128 then
      .bits (input.value (start+j.val) feature.val) else zero)))

/-- Explicit fixed-instruction causal attention recurrence, with the actual reduction
trees, replica ownership, cast locations and FMA carry order. -/
def step (length queryRow : Nat) (input : Inputs) (start : Nat) (old : Row) : Row :=
  ⟨maximum length queryRow start input old,
   fun owner=>.fma32 (old.normalizer owner) (alpha length queryRow start input old owner)
     ((plan owner).value Graph.add32
       (fun j=>probability length queryRow start input old (physicalOwner j.val) j)),
   fun feature owner=>.fma32 (old.value feature owner) (alpha length queryRow start input old owner)
     (contribution length queryRow start input old feature)⟩

def attention (length queryRow : Nat) (input : Inputs) (feature : Fin 128) : Graph :=
  let prefixState:=foldRange (step length queryRow input) 0 (queryRow/128*128) 64 initial
  let result:=foldRange (step length queryRow input) (queryRow/128*128)
    (min ((queryRow/128+1)*128) length) 64 prefixState
  .narrow16 (.divFull32 (result.value feature (physicalOwner feature.val))
    (result.normalizer (physicalOwner feature.val)))

/-- The only adapter reads initial Tensor words through the passed strides.
The recurrence above has no State or source-model argument. -/
def inputs (p : Source.Parameters) (initial : State Word) (batch head : Nat) : Inputs :=
  ⟨fun n d=>(initial.mem p.Q (batch*p.q.batch+head*p.q.head+n*p.q.row+d*p.q.dim)).read .real,
   fun n d=>(initial.mem p.K (batch*p.k.batch+head*p.k.head+d*p.k.dim+n*p.k.row)).read .real,
   fun n d=>(initial.mem p.V (batch*p.v.batch+head*p.v.head+n*p.v.row+d*p.v.dim)).read .real,
   fun block=>(initial.mem p.QS ((batch*p.heads+head)*((p.length+127)/128)+block)).read .real,
   fun block=>(initial.mem p.KS ((batch*p.heads+head)*((p.length+63)/64)+block)).read .real⟩

def target (instructions : Native.Instructions) (p : Source.Parameters)
    (initial : State Word) (batch head n d : Nat) (hd : d<128) : Word :=
  (attention p.length n (inputs p initial batch head) ⟨d,by omega⟩).eval instructions.ops

def ByteResult (instructions : Native.Instructions) (p : Source.Parameters)
    (storage : Whole.TensorStorage p) (placement : Whole.Placement p storage)
    (initial : State Word) (before after : Fin placement.arena → BitVec 8) : Prop :=
  (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),∀ k<2,∀ i : Fin placement.arena,
    i.val=placement.base .output+2*Whole.logicalAddress p b h n d+k →
      after i=Whole.byte (BitVec.ofNat 16 (target instructions p initial b h n d hd).toNat) k) ∧
  (∀ i : Fin placement.arena,(∀ off,Whole.Footprint p off → ∀ k<2,
    i.val≠placement.base .output+2*off+k) → after i=before i)

#specNonCircular attention avoiding [Whole.answer,Source.accumulated,Source.update,Whole.runPrograms,FP.Structural.exec]
#specNonCircular target avoiding [Whole.answer,Source.accumulated,Source.update,Whole.runPrograms,FP.Structural.exec]
#specNonCircular ByteResult avoiding [Whole.answer,Source.accumulated,Source.update,Whole.runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.NativeReference

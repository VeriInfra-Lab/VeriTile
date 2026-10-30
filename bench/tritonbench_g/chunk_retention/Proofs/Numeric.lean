import bench.tritonbench_g.chunk_retention.Sources.Whole
import bench.tritonbench_g.chunk_retention.Proofs.Transport
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
namespace Numeric
open VeriTile.Triton.FP
abbrev Word := BitVec 32

def value (x : Word) : Value .fp32 := ⟨x⟩
def observe (x : Word) : Word := if (value x).isNaN then (Value.qNaN .fp32).bits else x

def integer64 (x : BitVec 64) : Word := (round .fp32 .ieeeValue (Rat.ofInt x.toInt)).bits

def add32 (x y : Word) : Word := observe (add .ieeeValue (value (observe x)) (value (observe y))).bits
def sub32 (x y : Word) : Word := observe (sub .ieeeValue (value (observe x)) (value (observe y))).bits
def mul32 (x y : Word) : Word := observe (mul .ieeeValue (value (observe x)) (value (observe y))).bits
def fma32 (x y z : Word) : Word := observe (fma .ieeeValue (value (observe x))
  (value (observe y)) (value (observe z))).bits

/-- Fixed unary source profiles; neither function accepts a shape, buffer,
loop state or numerical result. hostScale names Python pow(K,-0.5) plus FP32 ABI.
The unused zero K case is totalized; the host scope always requires K>0. -/
structure UnaryModels where
  exp2FTZ : Word → Word
  log2Device : Word → Word
  hostScale : BitVec 32 → Word

def referenceISA (u : UnaryModels) : ISA where
  integer := fun n => integer64 (BitVec.ofInt 64 n)
  add := add32
  sub := sub32
  mul := mul32
  fma := fma32
  exp2 := fun x => observe (u.exp2FTZ (observe x))
  log2 := fun x => observe (u.log2Device (observe x))

/-- Every native numerical port has a fixed bit width and arity at most three. -/
structure ScalarPorts where
  integer64 : BitVec 64 → Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
  exp2FTZ : Word → Word
  log2Device : Word → Word
  hostScale : BitVec 32 → Word

structure ScalarLaws (u : UnaryModels) (p : ScalarPorts) : Prop where
  integer64 : ∀ n,observe (p.integer64 n)=integer64 n
  add : ∀ x y,observe (p.add x y)=add32 x y
  sub : ∀ x y,observe (p.sub x y)=sub32 x y
  mul : ∀ x y,observe (p.mul x y)=mul32 x y
  fma : ∀ x y z,observe (p.fma x y z)=fma32 x y z
  exp2FTZ : ∀ x,observe (p.exp2FTZ x)=observe (u.exp2FTZ (observe x))
  log2Device : ∀ x,observe (p.log2Device x)=observe (u.log2Device (observe x))
  hostScale : ∀ k,observe (p.hostScale k)=observe (u.hostScale k)

def nativeISA (p : ScalarPorts) : ISA where
  integer := fun n => observe (p.integer64 (BitVec.ofInt 64 n))
  add := fun x y => observe (p.add x y)
  sub := fun x y => observe (p.sub x y)
  mul := fun x y => observe (p.mul x y)
  fma := fun x y z => observe (p.fma x y z)
  exp2 := fun x => observe (p.exp2FTZ x)
  log2 := fun x => observe (p.log2Device x)

theorem isa_eq (u : UnaryModels) (p : ScalarPorts) (laws : ScalarLaws u p) :
    nativeISA p=referenceISA u := by
  have hi : (fun n => observe (p.integer64 (BitVec.ofInt 64 n)))=
      (fun n => integer64 (BitVec.ofInt 64 n)) := funext (fun n=>laws.integer64 _)
  have ha := funext (fun x=>funext (laws.add x))
  have hs := funext (fun x=>funext (laws.sub x))
  have hm := funext (fun x=>funext (laws.mul x))
  have hf := funext (fun x=>funext (fun y=>funext (laws.fma x y)))
  have he := funext laws.exp2FTZ
  have hl := funext laws.log2Device
  simp only [nativeISA,referenceISA,hi,ha,hs,hm,hf,he,hl]

def referenceRun (u : UnaryModels) (c : Shape) (schedule : SumTree) (flag : Bool)
    (entry : Heap Word) : Heap Word :=
  Source.run (wordOps (referenceISA u)) c (observe (u.hostScale (BitVec.ofNat 32 c.K))) schedule flag entry

def nativeRun (p : ScalarPorts) (c : Shape) (schedule : SumTree) (flag : Bool)
    (entry : Heap Word) : Heap Word :=
  Source.run (wordOps (nativeISA p)) c (observe (p.hostScale (BitVec.ofNat 32 c.K))) schedule flag entry

/-- Complete current-memory source execution composition. The only numerical
premises are the eight fixed scalar laws. Source/compiler scalar order, legal
launch completion, tensor metadata and physical memory binding are ordinary
structural contracts; there is no program result premise. -/
theorem native_eq (u : UnaryModels) (p : ScalarPorts) (laws : ScalarLaws u p)
    (c : Shape) (schedule : SumTree) (flag : Bool) (entry : Heap Word) :
    nativeRun p c schedule flag entry=referenceRun u c schedule flag entry := by
  simp only [nativeRun,referenceRun,isa_eq u p laws,laws.hostScale]

/-- Raw caller-frame equality is independent of numerical conformance. -/
theorem native_frame (p : ScalarPorts) (c : Shape) (schedule : SumTree) (flag : Bool)
    (entry : Heap Word) (point : Address) (outside : point∉Source.footprint c flag) :
    nativeRun p c schedule flag entry point=entry point :=
  Source.run_frame _ _ _ _ _ _ _ outside

#axiomsClean isa_eq
#axiomsClean native_eq
#axiomsClean native_frame
#specNonCircular ScalarLaws avoiding [nativeRun,referenceRun,Source.run,Reference.output]
end Numeric
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001

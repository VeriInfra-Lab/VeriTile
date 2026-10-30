import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Physical
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Bounds
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Host
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
noncomputable section
structure Tensor where
  identity : RegionName
  rows : Nat
  columns : Nat
  device : Nat
  deriving DecidableEq, Repr
structure Result where
  memory : Physical.Bytes
  output : Tensor
  returned : Option Tensor
  allocated : Bool
/-- On the allocating route, allocator bytes may be arbitrary. The output
has the documented half dtype on A's device; inputs remain on their caller
allocations. Successful public allocation/device legality are API boundaries. -/
def prepare (c : Call) (initial : Physical.Bytes) (fresh : Nat → Physical.Byte) : Physical.Bytes :=
  if c.supplied then initial else fun r b => if r="C" then fresh b else initial r b

def tensor (c : Call) (device : Nat) : Tensor := ⟨"C",c.M,c.N,device⟩
def call (c : Call) (device : Nat) (actual : Native.Instructions) (i : Fin 17)
    (initial : Physical.Bytes) (fresh : Nat → Physical.Byte) : Option Result :=
  (Physical.source actual c i (prepare c initial fresh)).map (fun memory =>
    ⟨memory,tensor c device,if c.supplied then none else some (tensor c device),!c.supplied⟩)

def Claim (mma : Opcode → ISA.MMA16) (c : Call) (device : Nat) (i : Fin 17)
    (initial : Physical.Bytes) (fresh : Nat → Physical.Byte) (result : Result) : Prop :=
  Physical.Post mma c i (prepare c initial fresh) result.memory ∧
  result.output=tensor c device ∧
  result.returned=(if c.supplied then none else some (tensor c device)) ∧
  result.allocated=(!c.supplied) ∧
  (∀ (r : RegionName) (b : Nat),r≠"C" → result.memory r b=initial r b)

theorem prepare_frame (c : Call) (initial : Physical.Bytes) (fresh : Nat → Physical.Byte)
    (r : RegionName) (b : Nat) (h : r≠"C") : prepare c initial fresh r b=initial r b := by
  cases hs : c.supplied <;> simp [prepare,hs,h]

theorem whole_call (c : Call) (hc : c.Valid) (device : Nat) (actual : Native.Instructions)
    (mma : Opcode → ISA.MMA16) (ports : Native.Obligations actual mma) (i : Fin 17)
    (initial : Physical.Bytes) (fresh : Nat → Physical.Byte) :
    ∃ result,call c device actual i initial fresh=some result ∧ Claim mma c device i initial fresh result := by
  obtain ⟨memory,he,hr,hf⟩ := Physical.whole_bytes actual mma ports c hc i (prepare c initial fresh)
  refine ⟨⟨memory,tensor c device,if c.supplied then none else some (tensor c device),!c.supplied⟩,?_,
    ⟨hr,hf⟩,rfl,rfl,rfl,?_⟩
  · simp only [call,he,Option.map_some]
  · intro r b h
    exact (hf r b (Or.inl h)).trans (prepare_frame c initial fresh r b h)

def software (mma : Opcode → ISA.MMA16) : Native.Instructions where
  mul := MatmulDequantInt4.Binary16.mul .ieeeValue
  fma := MatmulDequantizeInt4.FPCompiled.HalfFMA.fma .ieeeValue
  narrow := QuantizeKvTransform.MixedBits.narrow .ieeeValue
  mma := mma

theorem software_ports (mma : Opcode → ISA.MMA16) : Native.Obligations (software mma) mma :=
  ⟨fun _ _ => rfl,fun _ _ _ => rfl,fun _ => rfl,fun _ _ _ _ => rfl⟩

theorem supplied_returns_none (c : Call) (device : Nat) (mma : Opcode → ISA.MMA16)
    (i : Fin 17) (m : Physical.Bytes) (fresh : Nat → Physical.Byte) (r : Result)
    (h : Claim mma c device i m fresh r) (hs : c.supplied=Bool.true) :
    r.returned=none ∧ r.allocated=Bool.false := by
  simpa only [hs,Bool.not_true,ite_true] using And.intro h.2.2.1 h.2.2.2.1

theorem allocated_returns_output (c : Call) (device : Nat) (mma : Opcode → ISA.MMA16)
    (i : Fin 17) (m : Physical.Bytes) (fresh : Nat → Physical.Byte) (r : Result)
    (h : Claim mma c device i m fresh r) (hs : c.supplied=Bool.false) :
    r.returned=some r.output ∧ r.allocated=Bool.true := by
  rw [h.2.1]
  simpa only [hs,Bool.not_false,ite_false] using And.intro h.2.2.1 h.2.2.2.1

#specNonCircular Claim avoiding [call,Physical.source,Native.grid,Grid.grid]
#axiomsClean whole_call
#axiomsClean software_ports
#axiomsClean supplied_returns_none
#axiomsClean allocated_returns_output
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Host

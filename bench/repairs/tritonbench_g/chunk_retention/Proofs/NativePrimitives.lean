import bench.repairs.tritonbench_g.chunk_retention.Proofs.CheckedBinding

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
/-- Fixed raw-word local instructions, including source host pow and ABI
marshalling. No matrix/reduction/stage/output oracle is a primitive. -/
abbrev Primitives := Numeric.ScalarPorts

def rawISA (p : Primitives) : ISA :=
  ⟨fun n=>p.integer64 (BitVec.ofInt 64 n),p.add,p.sub,p.mul,p.fma,p.exp2FTZ,p.log2Device⟩
def nativeOps (p : Primitives) : Ops Word := wordOps (rawISA p)
def nativeScale (p : Primitives) (c : Shape) := p.hostScale (BitVec.ofNat 32 c.K)

/-- Exact raw-word laws are distinct from the retained historical API that
observes/canonicalizes NaNs. Structural safety never uses these laws. -/
structure LocalLaws (a b : Primitives) : Prop where
  integer64 : ∀ n,a.integer64 n=b.integer64 n
  add : ∀ x y,a.add x y=b.add x y
  sub : ∀ x y,a.sub x y=b.sub x y
  mul : ∀ x y,a.mul x y=b.mul x y
  fma : ∀ x y z,a.fma x y z=b.fma x y z
  exp2FTZ : ∀ x,a.exp2FTZ x=b.exp2FTZ x
  log2Device : ∀ x,a.log2Device x=b.log2Device x
  hostScale : ∀ k,a.hostScale k=b.hostScale k

theorem local_eq (a b : Primitives) (h : LocalLaws a b) : a=b := by
  cases a;cases b
  have h1:=funext h.integer64
  have h2:=funext (fun x=>funext (h.add x))
  have h3:=funext (fun x=>funext (h.sub x))
  have h4:=funext (fun x=>funext (h.mul x))
  have h5:=funext (fun x=>funext (fun y=>funext (h.fma x y)))
  have h6:=funext h.exp2FTZ
  have h7:=funext h.log2Device
  have h8:=funext h.hostScale
  congr

theorem local_witness (p : Primitives) : LocalLaws p p := by constructor <;> intros <;> rfl

#axiomsClean local_eq
#axiomsClean local_witness
#specNonCircular LocalLaws avoiding [Source.run,Reference.output,Reference.dq,Reference.dk,Reference.dv]
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked

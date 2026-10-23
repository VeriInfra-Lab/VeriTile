import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeStaging
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
abbrev Word := BitVec 32

structure Instructions where
  ops : ISA Word
  bitsIdentity : ∀ w,ops.bits w=w

/-- Numerical agreements concern only scalar instructions and one output
coordinate of the fixed m16n8k8 TF32 instruction. -/
structure InstructionAgreement (A B : Instructions) : Prop where
  literal32 : ∀ x,A.ops.literal32 x=B.ops.literal32 x
  add32 : ∀ a b,A.ops.add32 a b=B.ops.add32 a b
  sub32 : ∀ a b,A.ops.sub32 a b=B.ops.sub32 a b
  mul32 : ∀ a b,A.ops.mul32 a b=B.ops.mul32 a b
  exp2Approx : ∀ ftz a,A.ops.exp2Approx ftz a=B.ops.exp2Approx ftz a
  neg32 : ∀ a,A.ops.neg32 a=B.ops.neg32 a
  fma32 : ∀ a b c,A.ops.fma32 a b c=B.ops.fma32 a b c
  mmaTF32 : ∀ r c a b seed,A.ops.mmaTF32 r c a b seed=B.ops.mmaTF32 r c a b seed

def InstructionAgreement.transport (A B : Instructions) (h : InstructionAgreement A B) :
    LocalTransport id A.ops B.ops where
  bits:=by intro w;simp [A.bitsIdentity,B.bitsIdentity]
  literal32:=h.literal32
  add32:=h.add32
  sub32:=h.sub32
  mul32:=h.mul32
  exp2Approx:=h.exp2Approx
  neg32:=h.neg32
  fma32:=h.fma32
  mmaTF32:=h.mmaTF32

def inject : (d : TileDType) → Value Word d → Value Stage d
  | .real,w | .fp32,w | .fp16,w | .bf16,w | .f8e4,w | .f8e5,w=>.ready (.bits w)
  | .nat,n=>n | .int,n=>n | .bool,n=>n | .ptr,n=>n | .blockPtr,n=>n

def injectState (s : State Word) : State Stage where
  mem:=fun r o=>match s.mem r o with | .mk d v=>.mk d (inject d v)
  regs:=fun _ _ _=>none
  pids:=s.pids
  numPids:=s.numPids
  undef:=fun d r o=>inject d (s.undef d r o)

def injectAllocation (before : State Word) (a : Typed.Allocation before) :
    Typed.Allocation (injectState before) where
  state:=injectState a.state
  preserved:=by
    intro r o hr
    change (match a.state.mem r o with | .mk d v=>(Cell.mk d (inject d v) : Cell Stage))=_
    rw [a.preserved r o hr]
    rfl

def project (I : Instructions) (p : Descriptor) (runtime : Runtime) : (d : TileDType) → Value Stage d → Value Word d
  | .real,g | .fp32,g | .fp16,g | .bf16,g | .f8e4,g | .f8e5,g=>(lower p runtime g).eval I.ops
  | .nat,n=>n | .int,n=>n | .bool,n=>n | .ptr,n=>n | .blockPtr,n=>n

def projectCell (I : Instructions) (p : Descriptor) (runtime : Runtime) : Cell Stage → Cell Word
  | .mk d v=>.mk d (project I p runtime d v)

def projectState (I : Instructions) (p : Descriptor) (runtime : Runtime) (before : State Word) (after : State Stage) : State Word :=
  {before with mem:=fun r o=>projectCell I p runtime (after.mem r o)}

theorem project_inject (I : Instructions) (p : Descriptor) (runtime : Runtime) (d : TileDType) (v : Value Word d) :
    project I p runtime d (inject d v)=v := by
  cases d <;> simp [project,inject,lower,Graph.eval,I.bitsIdentity]

theorem injectState_project (I : Instructions) (p : Descriptor) (runtime : Runtime) (s : State Word) (r : RegionName) (o : Nat) :
    projectCell I p runtime ((injectState s).mem r o)=s.mem r o := by
  change projectCell I p runtime (match s.mem r o with | .mk d v=>.mk d (inject d v))=_
  generalize s.mem r o=cell
  cases cell with | mk d v=>simp only [projectCell,project_inject]

def encode : Cell Word → Word
  | .mk .real w | .mk .fp32 w | .mk .fp16 w | .mk .bf16 w | .mk .f8e4 w | .mk .f8e5 w=>w
  | _=>0

def graphEncode (I : Instructions) (p : Descriptor) (runtime : Runtime) (cell : Cell Stage) : Word :=
  encode (projectCell I p runtime cell)

theorem inject_bytes (I : Instructions) (p : Descriptor) (runtime : Runtime) (A : FlatAlloc) (raw : Nat → BitVec 8)
    (s : State Word) (bound : Typed.BytesBound A encode raw s) :
    Typed.BytesBound A (graphEncode I p runtime) raw (injectState s) := by
  intro i r o hi
  simpa only [graphEncode,injectState_project] using bound i r o hi

def outputQ (I : Instructions) (p : Descriptor) (runtime : Runtime) (c : Config) (s : State Word)
    (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : Word :=
  (referenceQ p runtime c (injectState s) h t e).eval I.ops
def outputK (I : Instructions) (p : Descriptor) (runtime : Runtime) (c : Config) (s : State Word)
    (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : Word :=
  (referenceK p runtime c (injectState s) h t e).eval I.ops
def outputG (I : Instructions) (p : Descriptor) (runtime : Runtime) (c : Config) (s : State Word)
    (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) : Word :=
  (referenceG p runtime c (injectState s) k h t).eval I.ops

#axiomsClean injectState_project
#axiomsClean inject_bytes
#specNonCircular outputQ avoiding [Typed.invoke,Typed.runGrid,FP.Structural.exec]
#specNonCircular outputK avoiding [Typed.invoke,Typed.runGrid,FP.Structural.exec]
#specNonCircular outputG avoiding [Typed.invoke,Typed.runGrid,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native

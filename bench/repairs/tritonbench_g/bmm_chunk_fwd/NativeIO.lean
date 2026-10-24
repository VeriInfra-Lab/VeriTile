import bench.repairs.tritonbench_g.bmm_chunk_fwd.IO
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeReferenceGrid
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelCandidate
import VeriTile.Triton.Float.ReadProgramIO
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option autoImplicit false

def coordinates (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) :=
  NativeReference.coordinatesFor (parameters p) c pid.toPids

def readMask (p : Parameters) (c : Host.Config) : Fin 3 → Nat → Prop
  | ⟨0,_⟩,off => p.shape.outputSize≠0 ∧ ∃ pid : Host.PID (parameters p) c,
      NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true ∧
      ∃ q : Fin (Host.cdiv p.shape.K c.K), ∃ i : Fin c.M, ∃ e : Fin c.K,
        SourceAccess.aMask (parameters p) c (coordinates p c pid) q i e=Bool.true ∧
        SourcePointers.aOffset (parameters p) c (coordinates p c pid) q i e=off
  | ⟨1,_⟩,off => p.shape.outputSize≠0 ∧ ∃ pid : Host.PID (parameters p) c,
      NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true ∧
      ∃ q : Fin (Host.cdiv p.shape.K c.K), ∃ e : Fin c.K, ∃ j : Fin c.N,
        SourceAccess.bMask (parameters p) c (coordinates p c pid) q e j=Bool.true ∧
        SourcePointers.bOffset (parameters p) c (coordinates p c pid) q e j=off
  | ⟨2,_⟩,off => p.shape.outputSize≠0 ∧ p.hasSeq=Bool.true ∧
      ∃ pid : Host.PID (parameters p) c,
        NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true ∧
        ((∃ i : Fin c.M, SourcePointers.row c (coordinates p c pid) i<SourcePointers.limit (parameters p) (coordinates p c pid) ∧
          SourcePointers.seqOffset (parameters p) (coordinates p c pid) (SourcePointers.row c (coordinates p c pid) i)=off) ∨
         (∃ j : Fin c.N, SourcePointers.col c (coordinates p c pid) j<SourcePointers.limit (parameters p) (coordinates p c pid) ∧
          SourcePointers.seqOffset (parameters p) (coordinates p c pid) (SourcePointers.col c (coordinates p c pid) j)=off))

def writeMask (p : Parameters) (c : Host.Config) (off : Nat) : Prop :=
  p.shape.outputSize≠0 ∧
    ((∃ pid : Host.PID (parameters p) c,
      NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true ∧
      ∃ i : Fin c.M, ∃ j : Fin c.N,
        SourceAccess.outMask (parameters p) c (coordinates p c pid) i j=Bool.true ∧
        SourcePointers.outOffset (parameters p) c (coordinates p c pid) i j=off) ∨
     (p.causal=Bool.true ∧ Cleanup.Upper (parameters p) off))

/-- Initial A/B mathematical carrier reads, signed sequence IDs, and arbitrary
initial FP32 output readback. The FP32 precision is retained in the source and
native expression backend; `.real` names the existing input Cell read view. -/
def inputs (p : Parameters) : List IOChannel :=
  [⟨.real,[dataExtent p],p.A⟩,⟨.real,[dataExtent p],p.B⟩,
   ⟨.int,[seqExtent p],Region.cast p.Seq⟩,⟨.fp32,[p.shape.outputSize],p.Out⟩]

def outputs (p : Parameters) : List IOChannel := [⟨.fp32,[p.shape.outputSize],p.Out⟩]

def memory (p : Parameters) (c : Host.Config) : MemoryIO Expr where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.A,p.B,Region.cast p.Seq,p.Out]
  read
    | ⟨0,_⟩,_,i | ⟨1,_⟩,_,i | ⟨2,_⟩,_,i | ⟨3,_⟩,_,i => i.1.val
  inputMask
    | ⟨0,_⟩,_,i => readMask p c 0 i.1.val
    | ⟨1,_⟩,_,i => readMask p c 1 i.1.val
    | ⟨2,_⟩,_,i => readMask p c 2 i.1.val
    | ⟨3,_⟩,_,_ => True
  write | ⟨0,_⟩,_,i=>i.1.val
  outputMask | ⟨0,_⟩,_,i=>writeMask p c i.1.val
  scratchWrite := fun k=>Fin.elim0 k
  scratchMask := fun k=>Fin.elim0 k

def mainInvocation (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (commuted : Bool)
    (pid : Host.PID (parameters p) c) : Invocation Expr where
  kernel := if commuted then Kernels.indexCommuted (parameters p) c else Kernels.fp32Source (parameters p) c
  backend := engine (backend rest c).algebra
  clearRegisters := Bool.true
  pids := pid.toPids
  numPids := some ((parameters p).grid c).toNumPids

def cleanupInvocation (rest : Algebra Expr) (p : Parameters)
    (pid : Cleanup.PID (parameters p)) : Invocation Expr where
  kernel := Kernels.clearUpper p.Out p.shape.chunk (p.shape.chunks*p.shape.groups)
  backend := engine (CleanupSource.algebra rest (.leaf .fp32 0))
  clearRegisters := Bool.true
  pids := pid.toPids
  numPids := some ((parameters p).grid Cleanup.config).toNumPids

/-- Original complete source order, preserving each actual predecessor memory
and the original fresh-register convention. Every Config remains admitted. -/
def schedule (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (commuted : Bool) : List (Invocation Expr) :=
  if p.shape.outputSize=0 then [] else
    (FPGrid.allPrograms (parameters p) c).map (mainInvocation rest p c commuted) ++
      if p.causal then (Finset.univ.toList : List (Cleanup.PID (parameters p))).map (cleanupInvocation rest p) else []

def io (rest : Algebra Expr) (p : Parameters) (c : Host.Config) : ReadProgramIO Expr Expr where
  toMemoryIO := memory p c
  schedule := fun _=>schedule rest p c Bool.false
  encode := id
  observe := fun _ _=>id
  resultRead | ⟨0,_⟩,_,i=>i.1.val
  resultMask := fun _ _ _=>True

def candidate (rest : Algebra Expr) (p : Parameters) (c : Host.Config) : ProgramIO.Variant (io rest p c).toProgramIO where
  schedule := fun _=>schedule rest p c Bool.true
  observe := fun _ _=>id

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native

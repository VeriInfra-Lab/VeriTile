import bench.tritonbench_g.bmm_chunk_fwd.NativeIO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option autoImplicit false

abbrev Values (p : Parameters) := ChannelValues Expr (inputs p)
abbrev Writes := Nat → Option Expr

def a (p : Parameters) (v : Values p) (off : Nat) : Expr :=
  if h : off<dataExtent p then v (0 : Fin 4) (⟨off,h⟩,PUnit.unit) else default

def b (p : Parameters) (v : Values p) (off : Nat) : Expr :=
  if h : off<dataExtent p then v (1 : Fin 4) (⟨off,h⟩,PUnit.unit) else default

def seq (p : Parameters) (v : Values p) (off : Nat) : Int :=
  if h : off<seqExtent p then v (2 : Fin 4) (⟨off,h⟩,PUnit.unit) else 0

def oldOutput (p : Parameters) (v : Values p) (off : Nat) : Expr :=
  if h : off<p.shape.outputSize then v (3 : Fin 4) (⟨off,h⟩,PUnit.unit) else default

/-- A previous program stores an FP32-tagged cell. If an input aliases that
cell, the source's real/int read view returns its existing mismatched-tag
default. No region-separation assumption is added to the original native scope. -/
def operands (p : Parameters) (v : Values p) (writes : Writes) : NativeReference.Inputs where
  a := fun off=>if p.A=p.Out ∧ (writes off).isSome then default else a p v off
  b := fun off=>if p.B=p.Out ∧ (writes off).isSome then default else b p v off
  seq := fun off=>if Region.cast p.Seq=p.Out ∧ (writes off).isSome then 0 else seq p v off

/-- Pure fixed-order native dot recurrence and masked scatter. Every tile's
values are computed from its entry arrays before any of its output stores. -/
def tile (p : Parameters) (c : Host.Config) (v : Values p)
    (pid : Host.PID (parameters p) c) (writes : Writes) : Writes :=
  let d := coordinates p c pid
  if NativeReference.activeBlock (parameters p) c d then
    (TileShape.allIndices [c.M,c.N]).foldl (fun acc i=>
      if SourceAccess.outMask (parameters p) c d i.1 i.2.1 then
        Function.update acc (SourcePointers.outOffset (parameters p) c d i.1 i.2.1)
          (some (NativeReference.output (parameters p) c d (operands p v writes) i))
      else acc) writes
  else writes

def grid (p : Parameters) (c : Host.Config) (v : Values p) : Writes :=
  (FPGrid.allPrograms (parameters p) c).foldl (fun current pid=>tile p c v pid current) (fun _=>none)

def complete (p : Parameters) (c : Host.Config) (v : Values p) : Writes :=
  if p.shape.outputSize=0 then fun _=>none else
    let main := grid p c v
    if p.causal then (Cleanup.addresses (parameters p)).foldl
      (fun current off=>Function.update current off (some (.leaf .fp32 0))) main else main

/-- Independent configured native recurrence, including the untouched initial
output cells of degenerate configurations and actual causal cleanup zeros. -/
def formula (p : Parameters) (c : Host.Config) (v : Values p) : ChannelValues Expr (outputs p) :=
  fun ⟨0,_⟩ i=>(complete p c v i.1.val).getD (oldOutput p v i.1.val)

#specNonCircular operands avoiding [FP.Structural.run,FP.Structural.exec,State.mem]
#specNonCircular tile avoiding [FP.Structural.run,FP.Structural.exec,State.mem]
#specNonCircular formula avoiding [FP.Structural.run,FP.Structural.exec,State.mem]
end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native

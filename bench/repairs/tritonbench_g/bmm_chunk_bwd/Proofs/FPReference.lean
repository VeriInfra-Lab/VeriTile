import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPDefault

/-! Independent default-FP32 blocked recurrence. FPOutput proves its full source
readback, and FPGridOutput composes all programs without numerical laws. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault
noncomputable section
variable {α : Type} [Inhabited α]

def row (s : State α) (BM BN : Nat) (p : RealProgram.Parameters) (i : Fin BM) : Nat :=
  s.pids 0 / ((p.K+BN-1)/BN)*BM+i.val

def column (s : State α) (BN : Nat) (p : RealProgram.Parameters) (j : Fin BN) : Nat :=
  s.pids 0 % ((p.K+BN-1)/BN)*BN+j.val

def chunk (s : State α) (p : RealProgram.Parameters) := s.pids 2 / p.groups

def head (s : State α) (p : RealProgram.Parameters) := s.pids 2 - chunk s p*p.groups

def length (s : State α) (p : RealProgram.Parameters) := min p.chunk (p.seqlen-chunk s p*p.chunk)

def dBase (s : State α) (p : RealProgram.Parameters) :=
  s.pids 1*p.dBatch+chunk s p*p.dChunk+head s p*p.dHead

def aBase (s : State α) (p : RealProgram.Parameters) :=
  s.pids 1*p.aBatch+chunk s p*p.chunk*p.aRow+head s p*p.aHead

def leftBlock (M : Algebra α) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (initial : State α) (step : Nat) : Values α .real [BM,BCS] := fun i =>
  M.cast (some .fp32) .real .real
    (if row initial BM BN p i.1 < p.chunk ∧ i.2.1.val < length initial p-step*BCS then
      (initial.mem p.Dout (dBase initial p + (row initial BM BN p i.1*p.dColumn+
        (step*BCS+i.2.1.val)*p.dRow))).read .real
    else M.literal (some .fp32) .real 0)

def rightBlock (M : Algebra α) (BN BCS : Nat) (p : RealProgram.Parameters)
    (initial : State α) (step : Nat) : Values α .real [BCS,BN] := fun i =>
  M.cast (some .fp32) .real .real
    (if i.1.val < length initial p-step*BCS ∧ column initial BN p i.2.1 < p.K then
      (initial.mem p.A (aBase initial p + ((step*BCS+i.1.val)*p.aRow+
        column initial BN p i.2.1*p.aColumn))).read .real
    else M.literal (some .fp32) .real 0)

/-- Ordered FP32 dot contribution, followed by a separate FP32 addition.
No local sum tree or fused accumulator is silently substituted. -/
def carry (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) : Nat → Values α .real [BM,BN]
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | step+1 => fun i => M.binary (some .fp32) .real .add
      (carry M D BM BN BCS p initial step i)
      (D.plain (some .fp32) (batch := []) (m := BM) (k := BCS) (n := BN) (leftBlock M BM BN BCS p initial step)
        (rightBlock M BN BCS p initial step) i)

def offset (s : State α) (BM BN : Nat) (p : RealProgram.Parameters)
    (sb ss sg sk : Nat) (i : TileIndex [BM,BN]) :=
  s.pids 1*sb+chunk s p*p.chunk*ss+head s p*sg+
    (row s BM BN p i.1*ss+column s BN p i.2.1*sk)

def expected (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) (i : TileIndex [BM,BN]) : α :=
  let acc := carry M D BM BN BCS p initial ((length initial p+BCS-1)/BCS) i
  let acc := if p.hasResidual then M.binary (some .fp32) .real .add acc
    (M.cast (some .fp32) .fp32 .real
      ((initial.mem p.Res (offset initial BM BN p p.rBatch p.rRow p.rHead p.rColumn i)).read .fp32))
    else acc
  M.cast (some .fp32) .real .fp32 acc

def active (s : State α) (BM BN : Nat) (p : RealProgram.Parameters) (i : TileIndex [BM,BN]) : Prop :=
  row s BM BN p i.1 < length s p ∧ column s BN p i.2.1 < p.K

/-- Independent source IO proposition. FPOutput.source_output proves it under
exact active-store injectivity; it is never an assumed numerical law. -/
def OutputGoal (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) : Prop :=
  ∃ final, FP.Structural.exec (engine (backend M D)) (source BM BN BCS p) initial = some final ∧
    (∀ i, active initial BM BN p i →
      final.mem p.Db (offset initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i) =
        .mk .fp32 (expected M D BM BN BCS p initial i)) ∧
    (∀ r a, (r ≠ p.Db ∨ ∀ i, active initial BM BN p i →
      a ≠ offset initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i) → final.mem r a = initial.mem r a)

#specNonCircular expected avoiding [FP.Structural.exec, FPDefault.source]
#specNonCircular carry avoiding [FP.Structural.exec, FPDefault.source]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPReference

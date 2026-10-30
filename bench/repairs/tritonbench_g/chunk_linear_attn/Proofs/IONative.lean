import bench.repairs.tritonbench_g.chunk_linear_attn.IO
import bench.repairs.tritonbench_g.chunk_linear_attn.NativeFormula
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.FiniteReferenceResult

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IONative
open VeriTile Triton FP.Structural FiniteReadLocality
noncomputable section
set_option maxHeartbeats 4000000

def canonical (c : KernelInterface.Config)
    (vals : ChannelValues (BitVec 32) (IOLayout.inputs c.dims)) : Memory NativeTerms.Term :=
  NativeTerms.inputMemory (NativeFormula.input c.dims vals)

/-- Finite shared IO binding supplies exactly the decoded reads used by the
old native arithmetic. The initial state's other cells, tags and undef remain arbitrary. -/
theorem inputs_agree (c : KernelInterface.Config)
    (vals : ChannelValues (BitVec 32) (IOLayout.inputs c.dims)) (before : State NativeTerms.Term)
    (binding : (IOLayout.nativeMemory (BitVec 32) c).InputBinding NativeTerms.Term.bits vals before) :
    InputsAgree c.dims c.seed c.terminal before.mem (canonical c vals) := by
  have point (i : Fin (IOLayout.inputs c.dims).length)
      (enabled : (i.val=4 → c.seed=Bool.true) ∧ (i.val=5 → c.terminal=Bool.true))
      (j : TileIndex (IOLayout.inputs c.dims)[i].shape) :
      (before.mem (IOLayout.inputs c.dims)[i].buf
        (TileShape.linearIndex (IOLayout.inputs c.dims)[i].shape j)).read .real =
      (canonical c vals (IOLayout.inputs c.dims)[i].buf
        (TileShape.linearIndex (IOLayout.inputs c.dims)[i].shape j)).read .real := by
    have bound := binding i j enabled
    fin_cases i
    · change (before.mem "query" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j)).read .real =
        NativeTerms.Term.bits (vals (0 : Fin 6) j) at bound
      change (before.mem "query" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j)).read .real =
        NativeTerms.Term.bits (NativeFormula.input c.dims vals "query" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j))
      simpa [NativeFormula.input,IOArray.read] using bound
    · change (before.mem "key" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j)).read .real =
        NativeTerms.Term.bits (vals (1 : Fin 6) j) at bound
      change (before.mem "key" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j)).read .real =
        NativeTerms.Term.bits (NativeFormula.input c.dims vals "key" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j))
      simpa [NativeFormula.input,IOArray.read] using bound
    · change (before.mem "value" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (vals (2 : Fin 6) j) at bound
      change (before.mem "value" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (NativeFormula.input c.dims vals "value" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j))
      simpa [NativeFormula.input,IOArray.read] using bound
    · change (before.mem "incoming" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (vals (3 : Fin 6) j) at bound
      change (before.mem "incoming" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (NativeFormula.input c.dims vals "incoming" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j))
      simpa [NativeFormula.input,IOArray.read] using bound
    · change (before.mem "initial" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (vals (4 : Fin 6) j) at bound
      change (before.mem "initial" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (NativeFormula.input c.dims vals "initial" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j))
      simpa [NativeFormula.input,IOArray.read] using bound
    · change (before.mem "terminalGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (vals (5 : Fin 6) j) at bound
      change (before.mem "terminalGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j)).read .real =
        NativeTerms.Term.bits (NativeFormula.input c.dims vals "terminalGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j))
      simpa [NativeFormula.input,IOArray.read] using bound
  have region (i : Fin (IOLayout.inputs c.dims).length)
      (enabled : (i.val=4 → c.seed=Bool.true) ∧ (i.val=5 → c.terminal=Bool.true)) :
      Reads (IOLayout.inputs c.dims)[i].buf (TileShape.product (IOLayout.inputs c.dims)[i].shape)
        before.mem (canonical c vals) := by
    intro off inside
    simpa only [IOArray.index_offset] using point i enabled (IOArray.index _ ⟨off,inside⟩)
  constructor
  · have h := region (0 : Fin 6) (by constructor <;> simp)
    change Reads "query" ((c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1))) before.mem (canonical c vals) at h
    simpa only [Nat.mul_one,Nat.mul_assoc] using h
  · have h := region (1 : Fin 6) (by constructor <;> simp)
    change Reads "key" ((c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1))) before.mem (canonical c vals) at h
    simpa only [Nat.mul_one,Nat.mul_assoc] using h
  · have h := region (2 : Fin 6) (by constructor <;> simp)
    change Reads "value" ((c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1))) before.mem (canonical c vals) at h
    simpa only [Nat.mul_one,Nat.mul_assoc] using h
  · have h := region (3 : Fin 6) (by constructor <;> simp)
    change Reads "incoming" ((c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1))) before.mem (canonical c vals) at h
    simpa only [Nat.mul_one,Nat.mul_assoc] using h
  · intro enabled
    have h := region (4 : Fin 6) ⟨fun _ => enabled,by simp⟩
    change Reads "initial" ((c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1))) before.mem (canonical c vals) at h
    simpa only [Nat.mul_one,Nat.mul_assoc] using h
  · intro enabled
    have h := region (5 : Fin 6) ⟨by simp,fun _ => enabled⟩
    change Reads "terminalGradient" ((c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1))) before.mem (canonical c vals) at h
    simpa only [Nat.mul_one,Nat.mul_assoc] using h

/-- The original complete four-grid and finite-sum result implements every
native IO output, including all written intermediates. No infinite input pin
or global storage-tag condition is introduced. -/
theorem outputs (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config)
    (valid : Host.Valid c.dims) (scale : BitVec 32) (work : KernelSchedule.Work)
    (vals : ChannelValues (BitVec 32) (IOLayout.inputs c.dims))
    (before after : State NativeTerms.Term)
    (binding : (IOLayout.nativeMemory (BitVec 32) c).InputBinding NativeTerms.Term.bits vals before)
    (memory : after.mem = completeMemory NativeTerms.algebra NativeTerms.dotPrimitive
      c.call.forward c.call.backward 1 c.seed c.final c.terminal c.call.selection before.mem) :
    (IO.native profile c scale work).OutputBinding vals (NativeFormula.native profile c valid scale vals) after := by
  let reference := KernelReference.result NativeTerms.algebra NativeTerms.dotPrimitive c.dims valid
    1 c.seed c.final c.terminal c.call.selection (canonical c vals)
  have same := result_agrees NativeTerms.algebra NativeTerms.dotPrimitive c.dims valid
    1 c.seed c.final c.terminal c.call.selection before.mem (canonical c vals) (inputs_agree c vals before binding)
  have typed : RealCells reference := result_real NativeTerms.algebra NativeTerms.dotPrimitive c.dims valid
    1 c.seed c.final c.terminal c.call.selection (canonical c vals) (fun _ _ => rfl)
  have target : after.mem = KernelReference.result NativeTerms.algebra NativeTerms.dotPrimitive c.dims valid
      1 c.seed c.final c.terminal c.call.selection before.mem :=
    memory.trans (KernelReference.result_eq NativeTerms.algebra NativeTerms.dotPrimitive c.dims valid
      1 c.seed c.final c.terminal c.call.selection before.mem)
  intro o j active
  have equal : after.mem (IOLayout.nativeOutputs c.dims)[o].buf
      (TileShape.linearIndex (IOLayout.nativeOutputs c.dims)[o].shape j) =
      reference (IOLayout.nativeOutputs c.dims)[o].buf
        (TileShape.linearIndex (IOLayout.nativeOutputs c.dims)[o].shape j) := by
    rw [target]
    fin_cases o
    · exact same.output _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j)
    · exact same.queryGradient _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j)
    · exact same.keyGradient _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.T,c.dims.K] j)
    · exact same.valueGradient _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.T,c.dims.V] j)
    · exact same.finalState (active.1 rfl) _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j)
    · exact same.initialGradient (active.2 rfl) _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.K,c.dims.V] j)
    · exact same.history _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.NT,c.dims.K,c.dims.V] j)
    · exact same.historyGradient _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [c.dims.B*c.dims.H,c.dims.NT,c.dims.K,c.dims.V] j)
    · exact same.valuePartials _ (by
        simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using TileShape.linearIndex_lt [keyBlocks (c.dims.parameters Bool.true),c.dims.B*c.dims.H,c.dims.T,c.dims.V] j)
  change mapCell _ (after.mem (IOLayout.nativeOutputs c.dims)[o].buf
    (TileShape.linearIndex (IOLayout.nativeOutputs c.dims)[o].shape j)) = _
  rw [equal,typed]
  fin_cases o <;> rfl

#axiomsClean inputs_agree
#axiomsClean outputs
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IONative

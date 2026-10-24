import bench.repairs.tritonbench_g.chunk_linear_attn.BoundMeaningGrid
import bench.repairs.tritonbench_g.chunk_linear_attn.HostIngress
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Host
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
noncomputable section
set_option maxHeartbeats 12000000

/-- Real interpretation uses the actual bound FP32 word, while the native
term retains the word itself, including the sign of zero. -/
theorem bound_graph_real (decode32 : BitVec 32 → ℝ) (decode64 : BitVec 64 → ℝ)
    (powerDefault : Nat → BitVec 64) (convert32 : BitVec 64 → BitVec 32) (c : Invocation)
    (abi : ScalarABI decode32 decode64 powerDefault convert32 c)
    (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    (NativeBound.graph c.scale.argumentFP32 c.mathCall bits r o).eval (NativeTerms.realISA decode32)=
      (c.mathCall.result (NativeTerms.decodedMemory decode32 bits) r o).read .real := by
  rw [NativeBound.graph_real,← abi.interpretation]
  rfl

def decodeWords (decode : BitVec 32 → ℝ) (physical : Nat → BitVec 32) : Nat → Cell ℝ :=
  fun o => .mk .real (decode (physical o))
structure WordCopies (c : Invocation) (a : Allocations) (physical : Nat → BitVec 32) : Prop where
  incoming : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V),
    physical (a.incomingDense.base+((b.val*c.dims.H+h.val)*c.dims.T+t.val)*c.dims.V+j.val)=
      match c.incoming with | none => (0 : BitVec 32) | some src => physical (src.base+offset4 src b.val h.val t.val j.val)
  terminal : ∀ src∈c.terminal, ∀ dst∈a.terminalDense,
    ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (e : Fin c.dims.K) (j : Fin c.dims.V),
      physical (dst.base+((b.val*c.dims.H+h.val)*c.dims.K+e.val)*c.dims.V+j.val)=
        physical (src.base+offset4 src b.val h.val e.val j.val)

theorem decode_copies (decode : BitVec 32 → ℝ) (zeroMeaning : decode 0=0)
    (c : Invocation) (a : Allocations) (physical : Nat → BitVec 32) (copies : WordCopies c a physical) :
    CotangentCopies Real.algebra c a (decodeWords decode physical) := by
  constructor
  · intro b h t j
    unfold decodeWords
    rw [copies.incoming]
    cases c.incoming <;> simp only [zeroMeaning]
    rfl
  · intro src hs dst hd b h e j
    unfold decodeWords
    rw [copies.terminal src hs dst hd b h e j]

#axiomsClean bound_graph_real
#axiomsClean decode_copies
#specNonCircular WordCopies avoiding [execute,Real.Call.result,NativeBound.graph]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Host

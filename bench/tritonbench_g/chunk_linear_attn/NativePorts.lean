import bench.tritonbench_g.chunk_linear_attn.NativeMeaningGrid
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

/-- Per-occurrence scalar numerical conditions. Each equality concerns one node
and its scalar operands. No whole dot, tensor, sum, state, or VJP answer is a port.
Unsupported unary, non-add/mul binary and MMA nodes are expressly excluded. -/
def LocalAgreement {α : Type} (f : α → ℝ) (decode : BitVec 32 → ℝ) (I : ISA α) : Graph → Prop
  | .bits b => f (I.bits b)=decode b
  | .literal p d x => f (I.literal p d x)=x
  | .binary p d op a b => LocalAgreement f decode I a ∧ LocalAgreement f decode I b ∧
      match op with
      | .add => f (I.binary p d op (a.eval I) (b.eval I))=f (a.eval I)+f (b.eval I)
      | .mul => f (I.binary p d op (a.eval I) (b.eval I))=f (a.eval I)*f (b.eval I)
      | _ => False
  | .cast p s t a => LocalAgreement f decode I a ∧ f (I.cast p s t (a.eval I))=f (a.eval I)
  | .fma a b c => LocalAgreement f decode I a ∧ LocalAgreement f decode I b ∧ LocalAgreement f decode I c ∧
      f (I.fma (a.eval I) (b.eval I) (c.eval I))=f (a.eval I)*f (b.eval I)+f (c.eval I)
  | .move _ a => LocalAgreement f decode I a
  | .unary .. | .mmaTF32 .. => False

theorem local_transport {α : Type} (f : α → ℝ) (decode : BitVec 32 → ℝ) (I : ISA α)
    (g : Graph) (h : LocalAgreement f decode I g) :
    f (g.eval I)=g.eval (realISA decode) := by
  induction g with
  | bits b => exact h
  | literal p d x => exact h
  | binary p d op a b ha hb =>
    rcases h with ⟨h1,h2,h3⟩
    cases op
    · change f (I.binary p d .add (a.eval I) (b.eval I))=a.eval (realISA decode)+b.eval (realISA decode)
      rw [h3,ha h1,hb h2]
    · exact False.elim h3
    · change f (I.binary p d .mul (a.eval I) (b.eval I))=a.eval (realISA decode)*b.eval (realISA decode)
      rw [h3,ha h1,hb h2]
    · exact False.elim h3
    · exact False.elim h3
    · exact False.elim h3
  | unary p op a ha => exact False.elim h
  | cast p s t a ha =>
    change f (I.cast p s t (a.eval I))=a.eval (realISA decode)
    exact h.2.trans (ha h.1)
  | fma a b c ha hb hc =>
    change f (I.fma (a.eval I) (b.eval I) (c.eval I))=
      a.eval (realISA decode)*b.eval (realISA decode)+c.eval (realISA decode)
    rw [h.2.2.2,ha h.1,hb h.2.1,hc h.2.2.1]
  | mmaTF32 lm ln a b c ha hb hc => exact False.elim h
  | move r a ha => exact ha h

theorem local_graph_real {α : Type} (f : α → ℝ) (decode : BitVec 32 → ℝ) (I : ISA α)
    (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat)
    (ports : LocalAgreement f decode I (graph c bits r o)) :
    f ((graph c bits r o).eval I)=(c.result (decodedMemory decode bits) r o).read .real := by
  rw [local_transport f decode I _ ports,graph_real]

#axiomsClean local_transport
#axiomsClean local_graph_real
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms

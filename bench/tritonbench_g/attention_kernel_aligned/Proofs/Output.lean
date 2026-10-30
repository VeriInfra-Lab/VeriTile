import bench.tritonbench_g.attention_kernel_aligned.Proofs.Complete
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Scatter
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
open VeriTile Triton FP.Structural Source Full FPValue
noncomputable section
variable {α : Type} [Inhabited α]
def active (p : Parameters) (s : State α) (i : QLane p) : Prop := row p s i.1<p.length
instance (p : Parameters) (s : State α) : DecidablePred (active p s) := fun _=>inferInstanceAs (Decidable (_<_))
def address (p : Parameters) (s : State α) := outputOffset p s
def answer (P : Primitives α) (p : Parameters) (s : State α) := outputValue P p s

theorem source_memory (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (s : State α) (hBN : 0<p.blockN) :
    ∃ t,FP.Structural.exec (backend rest P) (source p) s=some t ∧ t.mem=(written P p s).mem := by
  obtain ⟨t,he,hm,_⟩:=Source.correctness rest P p s hBN
  exact ⟨t,he,hm⟩

theorem readback (P : Primitives α) (p : Parameters) (s : State α)
    (i : QLane p) (hi : active p s i)
    (hinj : ∀ a b,active p s a → active p s b → address p s a=address p s b → a=b) :
    (written P p s).mem p.output (address p s i)=.mk .fp16 (answer P p s i) := by
  exact Scatter.masked_readback_active p.output (address p s)
    (fun j=>.mk .fp16 (answer P p s j)) (active p s) s i hi hinj

theorem frame (P : Primitives α) (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (ho : r≠p.output ∨ ∀ i,active p s i → o≠address p s i) :
    (written P p s).mem r o=s.mem r o := by
  exact State.masked_scatter_frame p.output (address p s) (fun j=>.mk .fp16 (answer P p s j))
    (active p s) (TileShape.allIndices _) r o (ho.imp id (fun h i _ hi=>h i hi)) s
#axiomsClean source_memory
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Whole

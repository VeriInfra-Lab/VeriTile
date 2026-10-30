import bench.repairs.tritonbench_g.context_attn_llama.Proofs.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
noncomputable section

/-- Independent finite-sentinel mathematical attention over arbitrary Real
payload arrays, gathered by the original request/page metadata. Padded keys
have zero values but retain their finite score in the denominator. -/
def mathematical (c : Call) (a : Formula.Inputs) (b h row dim : Nat) : ℝ :=
  (∑ j : Fin (count c.layout ⟨row/c.layout.BM,b*c.layout.heads+h⟩ a*c.layout.BN),
    Real.exp (Real.log 2 * score c.layout ⟨row/c.layout.BM,b*c.layout.heads+h⟩ a
      (Host.scale c.layout.D) (row%c.layout.BM) j.val) *
    value c.layout ⟨row/c.layout.BM,b*c.layout.heads+h⟩ a j.val dim) /
  ∑ j : Fin (count c.layout ⟨row/c.layout.BM,b*c.layout.heads+h⟩ a*c.layout.BN),
    Real.exp (Real.log 2 * score c.layout ⟨row/c.layout.BM,b*c.layout.heads+h⟩ a
      (Host.scale c.layout.D) (row%c.layout.BM) j.val)

theorem mathematical_eq (c : Call) (a : Formula.Inputs) (b h row dim : Nat) :
    mathematical c a b h row dim=Formula.expected c.layout
      ⟨row/c.layout.BM,b*c.layout.heads+h⟩ a (Host.scale c.layout.D) (row%c.layout.BM) dim := by
  simp only [mathematical,Formula.expected,Formula.numerator,Formula.denominator,Fin.is_lt,↓reduceIte]
  rfl

theorem whole_mathematical_real (c : Call) (d : Dimensions) (v : Views) (A : FlatAlloc)
    (s : BlockState) (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (checked : checkInputs c d v A.extent requestedMax start sequence prompt request pages=Bool.true)
    (observed : MetadataObservation c s start sequence prompt request pages) (labels : Labels c)
    (allocation : A.Disjoint) (covered : ∀ rg,rg∉A.regions → A.extent rg=0) :
    Spec.Real (∃ result,c.Exec A s Bool.false result ∧
      (∀ b : Fin c.batches,∀ h : Fin c.layout.heads,∀ row,row < c.length b → ∀ dim : Fin c.layout.D,
        result.2.readMem A.flat (A.addr c.regions.out (c.output b h row dim))=
          mathematical c (Program.inputs s c.regions) b h row dim) ∧
      BlockState.WriteWithin (c.writes A) (A.flattenState s) result.2 ∧
      (∀ rg∈c.inputs,∀ off,off < A.extent rg → result.2.mem A.flat (A.addr rg off)=
        (A.flattenState s).mem A.flat (A.addr rg off)) ∧ result.1=none) := by
  have guards := (checkInputs_iff c d v A.extent requestedMax start sequence prompt request pages).mp checked
  have hv := valid c d A s start sequence prompt request pages guards.2.1 guards.2.2.2 observed labels
  obtain ⟨result,run,values,frame,reads⟩ := whole_call c A s hv allocation covered Bool.false
  refine ⟨result,run,?_,frame,reads,run.1⟩
  intro b h row hr dim
  rw [mathematical_eq]
  exact values b h row hr (full_coverage c b ((guards.2.1.2.2.2.2.2.2.1 b).2.1) row hr) dim

#specNonCircular mathematical avoiding [Call.Exec,Call.Result,VeriTile.Triton.exec,Kernel.mergeFrames]
#axiomsClean whole_mathematical_real
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Checked

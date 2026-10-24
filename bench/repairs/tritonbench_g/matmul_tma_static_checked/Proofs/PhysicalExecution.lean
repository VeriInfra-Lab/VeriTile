import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.ByteExecution

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.PhysicalExecution
open VeriTile.Triton FP.Structural FPValue Host TensorABI Native Bytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Both complete tiles are loaded from the actual finite source pointers;
then the selected dot backend and output conversion run, before any store. -/
def values (M : Algebra Word) (D : Dot Word) (h : Shape) (f : Flags)
    (a b : TensorABI.Tensor) (read : Read) (i : CIndex (params h f)) : Word :=
  M.cast none .fp32 (outputDType f.half)
    (D.apply (some .fp32) {} (batch:=[])
      (fun j => readWord read Bool.true (Addresses.aPointer h f a j).toNat)
      (fun j => readWord read Bool.true (Addresses.bPointer h f b j).toNat) i)

def program (M : Algebra Word) (D : Dot Word) (h : Shape) (f : Flags)
    (a b c : TensorABI.Tensor) (t : ByteExecution.Machine) : ByteExecution.Machine :=
  (TileShape.allIndices [h.rows,h.columns]).foldl
    (ByteExecution.step h f c (values M D h f a b t.2)) t

theorem loads (read : Read) (h : Shape) (f : Flags) (dim : Addresses.Dimensions h)
    (a b c : TensorABI.Tensor) (abi : Binding h f a b c) (s : State Word)
    (input : Inputs (readWord read Bool.true) h a b s) :
    (∀ i, aValues (params h f) s i=readWord read Bool.true (Addresses.aPointer h f a i).toNat) ∧
    (∀ i, bValues (params h f) s i=readWord read Bool.true (Addresses.bPointer h f b i).toNat) := by
  have data := source_inputs (readWord read Bool.true) h f a b c abi s input
  have ptrs := Addresses.all_coordinates h f dim a b c abi
  exact ⟨fun i => by rw [ptrs.1 i]; exact data.1 i,
    fun i => by rw [ptrs.2.1 i]; exact data.2 i⟩

theorem values_eq (M : Algebra Word) (D : Dot Word) (read : Read) (h : Shape) (f : Flags)
    (dim : Addresses.Dimensions h) (a b c : TensorABI.Tensor) (abi : Binding h f a b c)
    (s : State Word) (input : Inputs (readWord read Bool.true) h a b s) :
    values M D h f a b read=outputValues M D (params h f) f.half s := by
  have data := loads read h f dim a b c abi s input
  funext i
  simp only [values,outputValues,←data.1,←data.2]

/-- Source physical load/dot/cast/store interpretation refines the shared DSL;
the proof connects each operand coordinate and byte store, not whole lowering. -/
theorem program_eq (M : Algebra Word) (D : Dot Word) (read : Read) (h : Shape) (f : Flags)
    (dim : Addresses.Dimensions h) (a b c : TensorABI.Tensor) (abi : Binding h f a b c)
    (s : State Word) (input : Inputs (readWord read Bool.true) h a b s) :
    program M D h f a b c (s,read)=ByteExecution.program M D h f c (s,read) := by
  unfold program ByteExecution.program
  rw [values_eq M D read h f dim a b c abi s input]

#axiomsClean loads
#axiomsClean program_eq
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.PhysicalExecution

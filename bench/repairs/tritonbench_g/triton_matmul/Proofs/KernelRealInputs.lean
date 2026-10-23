import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelRealModel
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelRealInputs
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter KernelRealModel
noncomputable section

/-- Arbitrary real values indexed by physical element-start address. This is an
ideal interpretation of memory, with no finite encoding or decode restriction. -/
abbrev RealRead := Nat → ℝ

def logicalValue (read : RealRead) (t : Tensor) (i : Nat) : ℝ :=
  read (physicalAddress (storage t.metadata) t i)

structure Inputs (read : RealRead) (a b : Tensor) (s : State ℝ) : Prop where
  left : ∀ i,i<a.metadata.numel → s.mem TensorBinding.inputA i=
    stored (storage a.metadata) (read (a.metadata.dataPtr.toNat+width (storage a.metadata)*i))
  right : ∀ i,i<b.metadata.numel → s.mem TensorBinding.inputB i=
    stored (storage b.metadata) (read (b.metadata.dataPtr.toNat+width (storage b.metadata)*i))

theorem direct_read (read : RealRead) (t : Tensor) (laws : Laws (storage t.metadata) t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    read (t.metadata.dataPtr.toNat+width (storage t.metadata)*i)=logicalValue read t i := by
  rw [laws.pointer (by omega)]
  simp only [logicalValue,physicalAddress,laws.contiguous contiguous i hi,Nat.mul_add,Nat.add_assoc]


def left (read : RealRead) (a b : Tensor) : Fin (rows a.metadata) → Fin (columns a.metadata) → ℝ :=
  fun r k => logicalValue read a (r.val*columns a.metadata+k.val)
def right (read : RealRead) (a b : Tensor) : Fin (columns a.metadata) → Fin (columns b.metadata) → ℝ :=
  fun k c => logicalValue read b (k.val*columns b.metadata+c.val)

theorem source_inputs (read : RealRead) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (s : State ℝ) (bound : Inputs read a b s) :
    (∀ r k,Repair.inputA (request a b) s r.val k.val=left read a b r k) ∧
    (∀ k c,Repair.inputB (request a b) s k.val c.val=right read a b k c) := by
  constructor
  · intro r k
    have hi : r.val*columns a.metadata+k.val<a.metadata.numel := by
      rw [matrix_numel a.metadata h.rank.1]
      exact linear_bound _ _ _ _ r.isLt k.isLt
    simp only [Repair.inputA,request,Nat.mul_one,left]
    rw [bound.left _ hi,direct_read read a la h.contiguous.1 _ hi,read_stored]
  · intro k c
    have hi : k.val*columns b.metadata+c.val<b.metadata.numel := by
      rw [matrix_numel b.metadata h.rank.2,←h.inner]
      exact linear_bound _ _ _ _ k.isLt c.isLt
    simp only [Repair.inputB,request,Nat.mul_one,right]
    rw [bound.right _ hi,direct_read read b lb h.contiguous.2 _ hi,
      ←same_storage a.metadata b.metadata h.dtype.2,read_stored]

/-- Independent full-K mathematical target on unrestricted real logical inputs. -/
def expected (read : RealRead) (a b : Tensor) (i : OutputIndex a b) : ℝ :=
  ∑ k : Fin (columns a.metadata),left read a b i.1 k*right read a b k i.2

theorem reference (read : RealRead) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (s : State ℝ) (bound : Inputs read a b s) (i : OutputIndex a b) :
    logicalReference idealNumerics (request a b) (left read a b) (right read a b) i.1 i.2=expected read a b i := by
  have input := source_inputs read a b la lb h s bound
  rw [←reference_logical idealNumerics (request a b) s _ _ input.1 input.2 i.1 i.2,
    ideal_reference (request a b) s _ _ input.1 input.2 i.1 i.2]
  rfl


#specNonCircular expected avoiding [Source.typed,FP.Structural.exec,FiniteGrid.grid]
#axiomsClean direct_read
#axiomsClean source_inputs
#axiomsClean reference
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelRealInputs

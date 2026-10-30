import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelExecution
import bench.tritonbench_g.quantize_copy_kv.Proofs.PtxSigned

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelNativeProfile
open VeriTile.Triton FP.Structural
namespace PtxSigned
export VeriTile.Bench.Repairs.QuantizeCopyKv.PtxSigned
  (cvtS16 byte signedByte CvtS16Spec cvtS16_spec cvtS16_unique agrees_on_defined_i8 boundary_examples)
end PtxSigned
abbrev Word := FP.Value .fp32
noncomputable section

structure RowPlan (width : Nat) where
  tree : Tree
  complete : tree.leaves.Perm (List.range width)
abbrev Plans := ∀ width,0<width → RowPlan width

/-- Only individual target instructions and complete ordered trees. A tree
is selected without the floating input values; no row/reduction oracle or
permutation-equivalence premise is admitted. -/
structure Arithmetic where
  absolute : Word → Word
  maxNode : Word → Word → Word
  divFull : Word → Word → Word
  equal : Word → Word → Bool
  plans : Plans

def zero : Word := FP.Value.ofBits .fp32 0
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def levels : Word := FP.Value.ofBits .fp32 0x42fe0000

def maximum (a : Arithmetic) {n : Nat} (values : Fin n → Word) : Word :=
  if hn : 0<n then (a.plans n hn).tree.evaluate a.maxNode
    (fun c => if hc : c<n then values ⟨c,hc⟩ else zero)
  else zero

def reduction (a : Arithmetic) {shape : TileShape} (ax : Fin shape.length) :
    (keep : Bool) → Values Word .real shape → Values Word .real (TileShape.reduceShape shape ax keep)
  | Bool.false,values => fun out => maximum a (fun c => values (TileShape.insertAxisIndex shape ax out c))
  | Bool.true,values => fun out => maximum a (fun c => values (TileShape.replaceAxisIndex shape ax out c))

def algebra (a : Arithmetic) (rest : Algebra Word) : Algebra Word :=
  { rest with
    literal := fun p d x => if x=0 then zero else if x=1 then one else if x=127 then levels else rest.literal p d x
    unary := fun p op x => match op with
      | .abs _ => a.absolute x
      | _ => rest.unary p op x
    binary := fun p d op x y => match op with
      | .div => a.divFull x y
      | _ => rest.binary p d op x y
    cast := fun _ _ _ x => x
    fp32Load := id
    reduceMax := fun _ => reduction a }

def backend (a : Arithmetic) (rest : Algebra Word) : Algebra Word :=
  KernelExecution.backend (algebra a rest) PtxSigned.signedByte a.equal

theorem reduction_complete (a : Arithmetic) (width : Nat) (hw : 0<width) :
    (a.plans width hw).tree.leaves.Perm (List.range width) := (a.plans width hw).complete

theorem source_literals : zero.toRat?=some 0 ∧ one.toRat?=some 1 ∧ levels.toRat?=some 127 := by
  decide +kernel

theorem conversion_callback (a : Arithmetic) (rest : Algebra Word) :
    (backend a rest).floatToInt (some .fp32) .real 8=
      some (fun x => some (PtxSigned.signedByte x)) := by simp [backend,KernelExecution.backend]

#axiomsClean reduction_complete
#axiomsClean source_literals
#axiomsClean conversion_callback
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelNativeProfile

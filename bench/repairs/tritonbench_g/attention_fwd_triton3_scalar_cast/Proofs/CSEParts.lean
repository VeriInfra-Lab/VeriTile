import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Reference

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section

def parts (c : Call) : Source.IntRange :=
  match Source.intRange? (Kernels.keyMaskCSE c).surfaceBody "start_n" with
  | some p => p
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split (c : Call) :
    (Kernels.keyMaskCSE c).surfaceBody=(parts c).before ++
      [.forRangeInt "start_n" (parts c).start (parts c).stop (parts c).stride (parts c).body] ++
      (parts c).after := rfl

@[simp] theorem before_eq (c : Call) : (parts c).before=(Reference.parts c).before := rfl
@[simp] theorem after_eq (c : Call) : (parts c).after=(Reference.parts c).after := rfl
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE

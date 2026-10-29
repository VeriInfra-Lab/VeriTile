import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeD16

/-! FP32 sm80 profile family for the four admitted widths. All use four
reduction replicas per row; D128 has eight warps and one M tile per warp.
The external structural audit checks each saved artifact independently. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativeProfiles
open VeriTile.Triton FP.Structural NativePhysical
noncomputable section

def warps (width : Nat) : Nat := if width ≤ 64 then 4 else 8

def qkOwner (width row column : Nat) : Owner :=
  row%(16*warps width)/16*32 + row%8*4 + column%8/2

def route (width : Nat) : Expr → Edge → Owner → Owner
  | _, .reductionInput column, owner => owner/4*4 + column%8/2
  | .dotAcc _ _ _ _ _ _ 128 128 _ _ _ (.product ..) _, .dotLeft linear, _ =>
      qkOwner width (linear/128) (linear%128)
  | _,_,owner => owner

def plan (width : Nat) : Plan where
  reverseMul := fun _ _ a b _ => NativeD16.reverseMul a b
  reverseFMA := fun _ _ _ _ _ => Bool.true
  sumTree := fun {_} _ _ _ owner => NativeD16.treeFor _ owner
  maxTree := fun {_} _ _ _ owner => NativeD16.treeFor _ owner
  route := route width

def ownerOf (width : Nat) : Expr → Owner
  | .dotAcc _ _ _ _ _ batch m _ n _ _ _ i =>
      let linear := TileShape.linearIndex (batch++[m,n]) i
      qkOwner width (linear/n) (linear%n)
  | _ => 0

theorem admitted_warps (width : Nat) (h : width = 16 ∨ width = 32 ∨ width = 64 ∨ width = 128) :
    warps width = 4 ∨ warps width = 8 := by
  rcases h with rfl | rfl | rfl | rfl <;> decide

#print axioms admitted_warps
end
end VeriTile.Bench.Repairs.ContextAttnNopad.NativeProfiles

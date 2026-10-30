import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeD16

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeProfiles
open VeriTile.Triton FP.Structural
open NativePhysical (Expr Owner Edge)
noncomputable section

def Supported (d : Nat) : Prop := d=16 ∨ d=32 ∨ d=64 ∨ d=128

/-- Exactly the wrapper's default launch choice. -/
def warps (d : Nat) : Nat := if d≤64 then 4 else 8

def producer (d row column : Nat) : Owner :=
  row%(warps d*16)/16*32 + row%8*4 + column%8/2

/-- Only PV has the source scaled-accumulator product. This distinguishes
the two 128x128 dots at D128 without inspecting any floating value. Q/K/V
leaves are raw input words; native loads remain identity word transport. -/
def route (d : Nat) : Expr → Edge → Owner → Owner
  | _, .reductionInput column, owner => owner/4*4 + column%8/2
  | .dotAcc _ _ _ _ _ _ 128 128 _ _ _ (.product ..) _, .dotLeft linear, _ =>
      producer d (linear/128) (linear%128)
  | _,_,owner => owner

def plan (d : Nat) : NativePhysical.Plan where
  reverseMul := fun _ _ a b _ => NativeD16.reverseMul a b
  reverseFMA := fun _ _ _ _ _ => Bool.true
  sumTree := fun {_} _ _ _ owner => NativeD16.treeFor _ owner
  maxTree := fun {_} _ _ _ owner => NativeD16.treeFor _ owner
  route := route d

def ownerOf (d : Nat) : Expr → Owner
  | .dotAcc _ _ _ _ _ batch m _ n _ _ _ i =>
      let linear := TileShape.linearIndex (batch++[m,n]) i
      producer d (linear/n) (linear%n)
  | _ => 0

theorem default_warps : warps 16=4 ∧ warps 32=4 ∧ warps 64=4 ∧ warps 128=8 := by decide

theorem all_reduction_replicas_complete (d owner : Nat) :
    ((plan d).sumTree (shape:=[128,128]) ⟨1,by decide⟩ Bool.false
      (⟨0,by decide⟩,PUnit.unit) owner).tree.indices.Perm (List.finRange 128) :=
  (NativeD16.treeFor 128 owner).complete

#axiomsClean default_warps
#axiomsClean all_reduction_replicas_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeProfiles

import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativePhysical

/-! Structural D16 profile. The external pinned audit checks these ordered
reduction trees against every PTX replica. No tree equality or numerical law
is used. The fallback only makes the interface total outside the selected
128-column reductions; it is not a claim about another compiler profile. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeD16
open VeriTile.Triton FP.Structural
open NativeGraph (Tree OrderedTree)
open NativePhysical (Expr Owner Edge)
noncomputable section
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def local0 : Tree 128 :=
  (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)) (.input ⟨8, by decide⟩)) (.input ⟨9, by decide⟩)) (.input ⟨16, by decide⟩)) (.input ⟨17, by decide⟩)) (.input ⟨24, by decide⟩)) (.input ⟨25, by decide⟩)) (.input ⟨32, by decide⟩)) (.input ⟨33, by decide⟩)) (.input ⟨40, by decide⟩)) (.input ⟨41, by decide⟩)) (.input ⟨48, by decide⟩)) (.input ⟨49, by decide⟩)) (.input ⟨56, by decide⟩)) (.input ⟨57, by decide⟩)) (.input ⟨64, by decide⟩)) (.input ⟨65, by decide⟩)) (.input ⟨72, by decide⟩)) (.input ⟨73, by decide⟩)) (.input ⟨80, by decide⟩)) (.input ⟨81, by decide⟩)) (.input ⟨88, by decide⟩)) (.input ⟨89, by decide⟩)) (.input ⟨96, by decide⟩)) (.input ⟨97, by decide⟩)) (.input ⟨104, by decide⟩)) (.input ⟨105, by decide⟩)) (.input ⟨112, by decide⟩)) (.input ⟨113, by decide⟩)) (.input ⟨120, by decide⟩)) (.input ⟨121, by decide⟩))

def local1 : Tree 128 :=
  (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨2, by decide⟩) (.input ⟨3, by decide⟩)) (.input ⟨10, by decide⟩)) (.input ⟨11, by decide⟩)) (.input ⟨18, by decide⟩)) (.input ⟨19, by decide⟩)) (.input ⟨26, by decide⟩)) (.input ⟨27, by decide⟩)) (.input ⟨34, by decide⟩)) (.input ⟨35, by decide⟩)) (.input ⟨42, by decide⟩)) (.input ⟨43, by decide⟩)) (.input ⟨50, by decide⟩)) (.input ⟨51, by decide⟩)) (.input ⟨58, by decide⟩)) (.input ⟨59, by decide⟩)) (.input ⟨66, by decide⟩)) (.input ⟨67, by decide⟩)) (.input ⟨74, by decide⟩)) (.input ⟨75, by decide⟩)) (.input ⟨82, by decide⟩)) (.input ⟨83, by decide⟩)) (.input ⟨90, by decide⟩)) (.input ⟨91, by decide⟩)) (.input ⟨98, by decide⟩)) (.input ⟨99, by decide⟩)) (.input ⟨106, by decide⟩)) (.input ⟨107, by decide⟩)) (.input ⟨114, by decide⟩)) (.input ⟨115, by decide⟩)) (.input ⟨122, by decide⟩)) (.input ⟨123, by decide⟩))

def local2 : Tree 128 :=
  (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨4, by decide⟩) (.input ⟨5, by decide⟩)) (.input ⟨12, by decide⟩)) (.input ⟨13, by decide⟩)) (.input ⟨20, by decide⟩)) (.input ⟨21, by decide⟩)) (.input ⟨28, by decide⟩)) (.input ⟨29, by decide⟩)) (.input ⟨36, by decide⟩)) (.input ⟨37, by decide⟩)) (.input ⟨44, by decide⟩)) (.input ⟨45, by decide⟩)) (.input ⟨52, by decide⟩)) (.input ⟨53, by decide⟩)) (.input ⟨60, by decide⟩)) (.input ⟨61, by decide⟩)) (.input ⟨68, by decide⟩)) (.input ⟨69, by decide⟩)) (.input ⟨76, by decide⟩)) (.input ⟨77, by decide⟩)) (.input ⟨84, by decide⟩)) (.input ⟨85, by decide⟩)) (.input ⟨92, by decide⟩)) (.input ⟨93, by decide⟩)) (.input ⟨100, by decide⟩)) (.input ⟨101, by decide⟩)) (.input ⟨108, by decide⟩)) (.input ⟨109, by decide⟩)) (.input ⟨116, by decide⟩)) (.input ⟨117, by decide⟩)) (.input ⟨124, by decide⟩)) (.input ⟨125, by decide⟩))

def local3 : Tree 128 :=
  (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨6, by decide⟩) (.input ⟨7, by decide⟩)) (.input ⟨14, by decide⟩)) (.input ⟨15, by decide⟩)) (.input ⟨22, by decide⟩)) (.input ⟨23, by decide⟩)) (.input ⟨30, by decide⟩)) (.input ⟨31, by decide⟩)) (.input ⟨38, by decide⟩)) (.input ⟨39, by decide⟩)) (.input ⟨46, by decide⟩)) (.input ⟨47, by decide⟩)) (.input ⟨54, by decide⟩)) (.input ⟨55, by decide⟩)) (.input ⟨62, by decide⟩)) (.input ⟨63, by decide⟩)) (.input ⟨70, by decide⟩)) (.input ⟨71, by decide⟩)) (.input ⟨78, by decide⟩)) (.input ⟨79, by decide⟩)) (.input ⟨86, by decide⟩)) (.input ⟨87, by decide⟩)) (.input ⟨94, by decide⟩)) (.input ⟨95, by decide⟩)) (.input ⟨102, by decide⟩)) (.input ⟨103, by decide⟩)) (.input ⟨110, by decide⟩)) (.input ⟨111, by decide⟩)) (.input ⟨118, by decide⟩)) (.input ⟨119, by decide⟩)) (.input ⟨126, by decide⟩)) (.input ⟨127, by decide⟩))

def tree0 : Tree 128 := .node (.node local0 local2) (.node local1 local3)

theorem complete0 : tree0.indices.Perm (List.finRange 128) := by decide

def tree1 : Tree 128 := .node (.node local1 local3) (.node local0 local2)

theorem complete1 : tree1.indices.Perm (List.finRange 128) := by decide

def tree2 : Tree 128 := .node (.node local2 local0) (.node local3 local1)

theorem complete2 : tree2.indices.Perm (List.finRange 128) := by decide

def tree3 : Tree 128 := .node (.node local3 local1) (.node local2 local0)

theorem complete3 : tree3.indices.Perm (List.finRange 128) := by decide

def ordered (owner : Owner) : OrderedTree 128 :=
  if owner%4=0 then ⟨tree0,complete0⟩ else
  if owner%4=1 then ⟨tree1,complete1⟩ else
  if owner%4=2 then ⟨tree2,complete2⟩ else ⟨tree3,complete3⟩

theorem distinct_lane_order : tree0.indices ≠ tree1.indices := by decide

def listTree {n : Nat} : List (Fin n) → Tree n
  | [] => .padding 0
  | x::xs => .node (.input x) (listTree xs)

theorem listTree_indices {n : Nat} (xs : List (Fin n)) : (listTree xs).indices = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [listTree,Tree.indices,ih]

def treeFor (n : Nat) (owner : Owner) : OrderedTree n :=
  if h : n=128 then h ▸ ordered owner
  else ⟨listTree (List.finRange n), by rw [listTree_indices]⟩

/-- Producer thread for the logical QK/P coordinate in the audited MMA layout. -/
def qkOwner (row column : Nat) : Owner := row%64/16*32 + row%8*4 + column%8/2

/-- Exact operand orientation selected by the audited source node shape. -/
def reverseMul : Expr → Expr → Bool
  | .dotAcc .., .literal .. => Bool.true
  | .unary .., .reduction .. => Bool.true
  | .binary _ _ .div .., .unary .. => Bool.true
  | _,_ => Bool.false

def route : Expr → Edge → Owner → Owner
  | _, .reductionInput column, owner => owner/4*4 + column%8/2
  | .dotAcc _ _ _ _ _ _ 128 128 16 _ _ _ _, .dotLeft linear, _ =>
      qkOwner (linear/128) (linear%128)
  | _,_,owner => owner

def plan : NativePhysical.Plan where
  reverseMul := fun _ _ a b _ => reverseMul a b
  reverseFMA := fun _ _ _ _ _ => Bool.true
  sumTree := fun {_} _ _ _ owner => treeFor _ owner
  maxTree := fun {_} _ _ _ owner => treeFor _ owner
  route := route

/-- The actual stored PV node retains its local row/channel. The final shared
layout conversion transports its word from precisely this producer. For an
empty loop the stored literal is owner-independent, and owner zero suffices. -/
def ownerOf : Expr → Owner
  | .dotAcc _ _ _ _ _ batch m _ n _ _ _ i =>
      let linear := TileShape.linearIndex (batch++[m,n]) i
      qkOwner (linear/n) (linear%n)
  | _ => 0

theorem treeFor_selected (owner : Owner) : treeFor 128 owner = ordered owner := by
  simp [treeFor]

#axiomsClean distinct_lane_order
#axiomsClean complete0
#axiomsClean complete1
#axiomsClean complete2
#axiomsClean complete3
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeD16

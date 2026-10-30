import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.NativeGraph

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSM90
open VeriTile.Triton FP.Structural NativeGraph
noncomputable section
set_option maxRecDepth 4096
set_option maxHeartbeats 4000000

def local64_0 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)) (.input ⟨8, by decide⟩)) (.input ⟨9, by decide⟩)) (.input ⟨16, by decide⟩)) (.input ⟨17, by decide⟩)) (.input ⟨24, by decide⟩)) (.input ⟨25, by decide⟩)) (.input ⟨32, by decide⟩)) (.input ⟨33, by decide⟩)) (.input ⟨40, by decide⟩)) (.input ⟨41, by decide⟩)) (.input ⟨48, by decide⟩)) (.input ⟨49, by decide⟩)) (.input ⟨56, by decide⟩)) (.input ⟨57, by decide⟩))

def local64_1 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨2, by decide⟩) (.input ⟨3, by decide⟩)) (.input ⟨10, by decide⟩)) (.input ⟨11, by decide⟩)) (.input ⟨18, by decide⟩)) (.input ⟨19, by decide⟩)) (.input ⟨26, by decide⟩)) (.input ⟨27, by decide⟩)) (.input ⟨34, by decide⟩)) (.input ⟨35, by decide⟩)) (.input ⟨42, by decide⟩)) (.input ⟨43, by decide⟩)) (.input ⟨50, by decide⟩)) (.input ⟨51, by decide⟩)) (.input ⟨58, by decide⟩)) (.input ⟨59, by decide⟩))

def local64_2 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨4, by decide⟩) (.input ⟨5, by decide⟩)) (.input ⟨12, by decide⟩)) (.input ⟨13, by decide⟩)) (.input ⟨20, by decide⟩)) (.input ⟨21, by decide⟩)) (.input ⟨28, by decide⟩)) (.input ⟨29, by decide⟩)) (.input ⟨36, by decide⟩)) (.input ⟨37, by decide⟩)) (.input ⟨44, by decide⟩)) (.input ⟨45, by decide⟩)) (.input ⟨52, by decide⟩)) (.input ⟨53, by decide⟩)) (.input ⟨60, by decide⟩)) (.input ⟨61, by decide⟩))

def local64_3 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨6, by decide⟩) (.input ⟨7, by decide⟩)) (.input ⟨14, by decide⟩)) (.input ⟨15, by decide⟩)) (.input ⟨22, by decide⟩)) (.input ⟨23, by decide⟩)) (.input ⟨30, by decide⟩)) (.input ⟨31, by decide⟩)) (.input ⟨38, by decide⟩)) (.input ⟨39, by decide⟩)) (.input ⟨46, by decide⟩)) (.input ⟨47, by decide⟩)) (.input ⟨54, by decide⟩)) (.input ⟨55, by decide⟩)) (.input ⟨62, by decide⟩)) (.input ⟨63, by decide⟩))

def tree64_0 : Tree 64 := .node (.node local64_0 local64_2) (.node local64_1 local64_3)
theorem complete64_0 : tree64_0.indices.Perm (List.finRange 64) := by decide

def tree64_1 : Tree 64 := .node (.node local64_1 local64_3) (.node local64_0 local64_2)
theorem complete64_1 : tree64_1.indices.Perm (List.finRange 64) := by decide

def tree64_2 : Tree 64 := .node (.node local64_2 local64_0) (.node local64_3 local64_1)
theorem complete64_2 : tree64_2.indices.Perm (List.finRange 64) := by decide

def tree64_3 : Tree 64 := .node (.node local64_3 local64_1) (.node local64_2 local64_0)
theorem complete64_3 : tree64_3.indices.Perm (List.finRange 64) := by decide

def ordered64 (owner : Owner) : OrderedTree 64 :=
  if owner%4=0 then ⟨tree64_0,complete64_0⟩ else
  if owner%4=1 then ⟨tree64_1,complete64_1⟩ else
  if owner%4=2 then ⟨tree64_2,complete64_2⟩ else ⟨tree64_3,complete64_3⟩

def local128_0 : Tree 128 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)) (.input ⟨8, by decide⟩)) (.input ⟨9, by decide⟩)) (.input ⟨16, by decide⟩)) (.input ⟨17, by decide⟩)) (.input ⟨24, by decide⟩)) (.input ⟨25, by decide⟩)) (.input ⟨32, by decide⟩)) (.input ⟨33, by decide⟩)) (.input ⟨40, by decide⟩)) (.input ⟨41, by decide⟩)) (.input ⟨48, by decide⟩)) (.input ⟨49, by decide⟩)) (.input ⟨56, by decide⟩)) (.input ⟨57, by decide⟩)) (.input ⟨64, by decide⟩)) (.input ⟨65, by decide⟩)) (.input ⟨72, by decide⟩)) (.input ⟨73, by decide⟩)) (.input ⟨80, by decide⟩)) (.input ⟨81, by decide⟩)) (.input ⟨88, by decide⟩)) (.input ⟨89, by decide⟩)) (.input ⟨96, by decide⟩)) (.input ⟨97, by decide⟩)) (.input ⟨104, by decide⟩)) (.input ⟨105, by decide⟩)) (.input ⟨112, by decide⟩)) (.input ⟨113, by decide⟩)) (.input ⟨120, by decide⟩)) (.input ⟨121, by decide⟩))

def local128_1 : Tree 128 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨2, by decide⟩) (.input ⟨3, by decide⟩)) (.input ⟨10, by decide⟩)) (.input ⟨11, by decide⟩)) (.input ⟨18, by decide⟩)) (.input ⟨19, by decide⟩)) (.input ⟨26, by decide⟩)) (.input ⟨27, by decide⟩)) (.input ⟨34, by decide⟩)) (.input ⟨35, by decide⟩)) (.input ⟨42, by decide⟩)) (.input ⟨43, by decide⟩)) (.input ⟨50, by decide⟩)) (.input ⟨51, by decide⟩)) (.input ⟨58, by decide⟩)) (.input ⟨59, by decide⟩)) (.input ⟨66, by decide⟩)) (.input ⟨67, by decide⟩)) (.input ⟨74, by decide⟩)) (.input ⟨75, by decide⟩)) (.input ⟨82, by decide⟩)) (.input ⟨83, by decide⟩)) (.input ⟨90, by decide⟩)) (.input ⟨91, by decide⟩)) (.input ⟨98, by decide⟩)) (.input ⟨99, by decide⟩)) (.input ⟨106, by decide⟩)) (.input ⟨107, by decide⟩)) (.input ⟨114, by decide⟩)) (.input ⟨115, by decide⟩)) (.input ⟨122, by decide⟩)) (.input ⟨123, by decide⟩))

def local128_2 : Tree 128 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨4, by decide⟩) (.input ⟨5, by decide⟩)) (.input ⟨12, by decide⟩)) (.input ⟨13, by decide⟩)) (.input ⟨20, by decide⟩)) (.input ⟨21, by decide⟩)) (.input ⟨28, by decide⟩)) (.input ⟨29, by decide⟩)) (.input ⟨36, by decide⟩)) (.input ⟨37, by decide⟩)) (.input ⟨44, by decide⟩)) (.input ⟨45, by decide⟩)) (.input ⟨52, by decide⟩)) (.input ⟨53, by decide⟩)) (.input ⟨60, by decide⟩)) (.input ⟨61, by decide⟩)) (.input ⟨68, by decide⟩)) (.input ⟨69, by decide⟩)) (.input ⟨76, by decide⟩)) (.input ⟨77, by decide⟩)) (.input ⟨84, by decide⟩)) (.input ⟨85, by decide⟩)) (.input ⟨92, by decide⟩)) (.input ⟨93, by decide⟩)) (.input ⟨100, by decide⟩)) (.input ⟨101, by decide⟩)) (.input ⟨108, by decide⟩)) (.input ⟨109, by decide⟩)) (.input ⟨116, by decide⟩)) (.input ⟨117, by decide⟩)) (.input ⟨124, by decide⟩)) (.input ⟨125, by decide⟩))

def local128_3 : Tree 128 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨6, by decide⟩) (.input ⟨7, by decide⟩)) (.input ⟨14, by decide⟩)) (.input ⟨15, by decide⟩)) (.input ⟨22, by decide⟩)) (.input ⟨23, by decide⟩)) (.input ⟨30, by decide⟩)) (.input ⟨31, by decide⟩)) (.input ⟨38, by decide⟩)) (.input ⟨39, by decide⟩)) (.input ⟨46, by decide⟩)) (.input ⟨47, by decide⟩)) (.input ⟨54, by decide⟩)) (.input ⟨55, by decide⟩)) (.input ⟨62, by decide⟩)) (.input ⟨63, by decide⟩)) (.input ⟨70, by decide⟩)) (.input ⟨71, by decide⟩)) (.input ⟨78, by decide⟩)) (.input ⟨79, by decide⟩)) (.input ⟨86, by decide⟩)) (.input ⟨87, by decide⟩)) (.input ⟨94, by decide⟩)) (.input ⟨95, by decide⟩)) (.input ⟨102, by decide⟩)) (.input ⟨103, by decide⟩)) (.input ⟨110, by decide⟩)) (.input ⟨111, by decide⟩)) (.input ⟨118, by decide⟩)) (.input ⟨119, by decide⟩)) (.input ⟨126, by decide⟩)) (.input ⟨127, by decide⟩))

def tree128_0 : Tree 128 := .node (.node local128_0 local128_2) (.node local128_1 local128_3)
theorem complete128_0 : tree128_0.indices.Perm (List.finRange 128) := by decide

def tree128_1 : Tree 128 := .node (.node local128_1 local128_3) (.node local128_0 local128_2)
theorem complete128_1 : tree128_1.indices.Perm (List.finRange 128) := by decide

def tree128_2 : Tree 128 := .node (.node local128_2 local128_0) (.node local128_3 local128_1)
theorem complete128_2 : tree128_2.indices.Perm (List.finRange 128) := by decide

def tree128_3 : Tree 128 := .node (.node local128_3 local128_1) (.node local128_2 local128_0)
theorem complete128_3 : tree128_3.indices.Perm (List.finRange 128) := by decide

def ordered128 (owner : Owner) : OrderedTree 128 :=
  if owner%4=0 then ⟨tree128_0,complete128_0⟩ else
  if owner%4=1 then ⟨tree128_1,complete128_1⟩ else
  if owner%4=2 then ⟨tree128_2,complete128_2⟩ else ⟨tree128_3,complete128_3⟩

def listTree {n : Nat} : List (Fin n) → Tree n
  | [] => .padding 0 | x::xs => .node (.input x) (listTree xs)
theorem listTree_indices {n : Nat} (xs : List (Fin n)) : (listTree xs).indices = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [listTree,Tree.indices,ih]

def treeFor (n : Nat) (owner : Owner) : OrderedTree n :=
  if h : n=64 then h ▸ ordered64 owner else
  if h : n=128 then h ▸ ordered128 owner else
  ⟨listTree (List.finRange n), by rw [listTree_indices]⟩

def qkOwner (row column : Nat) : Owner := row%64/16*32 + row%8*4 + column%8/2

/-- With eight warps and BM64, QK/P has two physical replica groups.
BM128 uses the second group for rows 64..127. The profile retains the group
chosen by the consumer instead of assuming equality between replicas. -/
def producer (bm d row column consumer : Nat) : Owner :=
  if d ≤ 64 then qkOwner row column else
  if bm = 128 then row%128/16*32 + row%8*4 + column%8/2
  else qkOwner row column + consumer/128*128

def storedOwner (bm d row column : Nat) : Owner :=
  if d ≤ 64 then qkOwner row column else
  if bm = 128 then row%128/16*32 + row%8*4 + column%8/2
  else qkOwner row column + column/(d/2)*128

def reverseMul : Expr → Expr → Bool
  | .dotAcc .., .literal .. => Bool.true
  | _,_ => Bool.false

def route (bm d : Nat) : Expr → Edge → Owner → Owner
  | _, .reductionInput column, owner => owner/4*4 + column%8/2
  | .dotAcc _ _ _ _ _ _ _ k _ _ _ (.product ..) _, .dotLeft linear, owner =>
      producer bm d (linear/k) (linear%k) owner
  | _,_,owner => owner

def widthFor (n : Nat) : MMAWidth :=
  if n=16 then .n16 else if n=32 then .n32 else if n=64 then .n64
  else if n=128 then .n128 else .n256

def plan (bm d : Nat) : NativeGraph.Plan where
  matrixWidth := fun n => widthFor (if bm=64 ∧ 64<d then n/2 else n)
  reverseMul := fun _ a b _ => reverseMul a b
  reverseFMA := fun _ _ _ _ => Bool.false
  sumTree := fun {_} _ _ _ owner => treeFor _ owner
  maxTree := fun {_} _ _ _ owner => treeFor _ owner
  route := route bm d

theorem tree64_selected (owner : Owner) : treeFor 64 owner = ordered64 owner := by simp [treeFor]
theorem tree128_selected (owner : Owner) : treeFor 128 owner = ordered128 owner := by simp [treeFor]
theorem distinct64 : tree64_0.indices ≠ tree64_1.indices := by decide
theorem distinct128 : tree128_0.indices ≠ tree128_1.indices := by decide

#print axioms complete128_3
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSM90

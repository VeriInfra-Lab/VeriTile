import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.NativeStaging

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelNative
open VeriTile.Triton Native
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

/- Fixed SM80 reduction trees. Each thread first reduces its adjacent pair,
then the later column pairs in increasing order. Warp shuffles use XOR 2,1.
The N-distributed eight-warp layout stores replica zero, reduces warps by
XOR 4,2,1, and broadcasts replica zero through shared memory. -/

def local64_0 : Tree 64 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)) (.input ⟨8, by decide⟩)) (.input ⟨9, by decide⟩)) (.input ⟨16, by decide⟩)) (.input ⟨17, by decide⟩)) (.input ⟨24, by decide⟩)) (.input ⟨25, by decide⟩)) (.input ⟨32, by decide⟩)) (.input ⟨33, by decide⟩)) (.input ⟨40, by decide⟩)) (.input ⟨41, by decide⟩)) (.input ⟨48, by decide⟩)) (.input ⟨49, by decide⟩)) (.input ⟨56, by decide⟩)) (.input ⟨57, by decide⟩)

def local64_1 : Tree 64 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨2, by decide⟩) (.input ⟨3, by decide⟩)) (.input ⟨10, by decide⟩)) (.input ⟨11, by decide⟩)) (.input ⟨18, by decide⟩)) (.input ⟨19, by decide⟩)) (.input ⟨26, by decide⟩)) (.input ⟨27, by decide⟩)) (.input ⟨34, by decide⟩)) (.input ⟨35, by decide⟩)) (.input ⟨42, by decide⟩)) (.input ⟨43, by decide⟩)) (.input ⟨50, by decide⟩)) (.input ⟨51, by decide⟩)) (.input ⟨58, by decide⟩)) (.input ⟨59, by decide⟩)

def local64_2 : Tree 64 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨4, by decide⟩) (.input ⟨5, by decide⟩)) (.input ⟨12, by decide⟩)) (.input ⟨13, by decide⟩)) (.input ⟨20, by decide⟩)) (.input ⟨21, by decide⟩)) (.input ⟨28, by decide⟩)) (.input ⟨29, by decide⟩)) (.input ⟨36, by decide⟩)) (.input ⟨37, by decide⟩)) (.input ⟨44, by decide⟩)) (.input ⟨45, by decide⟩)) (.input ⟨52, by decide⟩)) (.input ⟨53, by decide⟩)) (.input ⟨60, by decide⟩)) (.input ⟨61, by decide⟩)

def local64_3 : Tree 64 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨6, by decide⟩) (.input ⟨7, by decide⟩)) (.input ⟨14, by decide⟩)) (.input ⟨15, by decide⟩)) (.input ⟨22, by decide⟩)) (.input ⟨23, by decide⟩)) (.input ⟨30, by decide⟩)) (.input ⟨31, by decide⟩)) (.input ⟨38, by decide⟩)) (.input ⟨39, by decide⟩)) (.input ⟨46, by decide⟩)) (.input ⟨47, by decide⟩)) (.input ⟨54, by decide⟩)) (.input ⟨55, by decide⟩)) (.input ⟨62, by decide⟩)) (.input ⟨63, by decide⟩)

def tree64_0 : Tree 64 := .node (.node (local64_0) (local64_2)) (.node (local64_1) (local64_3))
theorem complete64_0 : tree64_0.leaves.Perm (List.finRange 64) := by decide

def tree64_1 : Tree 64 := .node (.node (local64_1) (local64_3)) (.node (local64_0) (local64_2))
theorem complete64_1 : tree64_1.leaves.Perm (List.finRange 64) := by decide

def tree64_2 : Tree 64 := .node (.node (local64_2) (local64_0)) (.node (local64_3) (local64_1))
theorem complete64_2 : tree64_2.leaves.Perm (List.finRange 64) := by decide

def tree64_3 : Tree 64 := .node (.node (local64_3) (local64_1)) (.node (local64_2) (local64_0))
theorem complete64_3 : tree64_3.leaves.Perm (List.finRange 64) := by decide

def warp64 (column : Nat) : OrderedTree 64 :=
  if column%8/2=0 then ⟨tree64_0,complete64_0⟩ else
  if column%8/2=1 then ⟨tree64_1,complete64_1⟩ else
  if column%8/2=2 then ⟨tree64_2,complete64_2⟩ else
  ⟨tree64_3,complete64_3⟩

def cross64_0 : Tree 64 :=
  .node (.node (.node (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)) (.node (.input ⟨4, by decide⟩) (.input ⟨5, by decide⟩))) (.node (.node (.input ⟨2, by decide⟩) (.input ⟨3, by decide⟩)) (.node (.input ⟨6, by decide⟩) (.input ⟨7, by decide⟩)))

def cross64_1 : Tree 64 :=
  .node (.node (.node (.input ⟨8, by decide⟩) (.input ⟨9, by decide⟩)) (.node (.input ⟨12, by decide⟩) (.input ⟨13, by decide⟩))) (.node (.node (.input ⟨10, by decide⟩) (.input ⟨11, by decide⟩)) (.node (.input ⟨14, by decide⟩) (.input ⟨15, by decide⟩)))

def cross64_2 : Tree 64 :=
  .node (.node (.node (.input ⟨16, by decide⟩) (.input ⟨17, by decide⟩)) (.node (.input ⟨20, by decide⟩) (.input ⟨21, by decide⟩))) (.node (.node (.input ⟨18, by decide⟩) (.input ⟨19, by decide⟩)) (.node (.input ⟨22, by decide⟩) (.input ⟨23, by decide⟩)))

def cross64_3 : Tree 64 :=
  .node (.node (.node (.input ⟨24, by decide⟩) (.input ⟨25, by decide⟩)) (.node (.input ⟨28, by decide⟩) (.input ⟨29, by decide⟩))) (.node (.node (.input ⟨26, by decide⟩) (.input ⟨27, by decide⟩)) (.node (.input ⟨30, by decide⟩) (.input ⟨31, by decide⟩)))

def cross64_4 : Tree 64 :=
  .node (.node (.node (.input ⟨32, by decide⟩) (.input ⟨33, by decide⟩)) (.node (.input ⟨36, by decide⟩) (.input ⟨37, by decide⟩))) (.node (.node (.input ⟨34, by decide⟩) (.input ⟨35, by decide⟩)) (.node (.input ⟨38, by decide⟩) (.input ⟨39, by decide⟩)))

def cross64_5 : Tree 64 :=
  .node (.node (.node (.input ⟨40, by decide⟩) (.input ⟨41, by decide⟩)) (.node (.input ⟨44, by decide⟩) (.input ⟨45, by decide⟩))) (.node (.node (.input ⟨42, by decide⟩) (.input ⟨43, by decide⟩)) (.node (.input ⟨46, by decide⟩) (.input ⟨47, by decide⟩)))

def cross64_6 : Tree 64 :=
  .node (.node (.node (.input ⟨48, by decide⟩) (.input ⟨49, by decide⟩)) (.node (.input ⟨52, by decide⟩) (.input ⟨53, by decide⟩))) (.node (.node (.input ⟨50, by decide⟩) (.input ⟨51, by decide⟩)) (.node (.input ⟨54, by decide⟩) (.input ⟨55, by decide⟩)))

def cross64_7 : Tree 64 :=
  .node (.node (.node (.input ⟨56, by decide⟩) (.input ⟨57, by decide⟩)) (.node (.input ⟨60, by decide⟩) (.input ⟨61, by decide⟩))) (.node (.node (.input ⟨58, by decide⟩) (.input ⟨59, by decide⟩)) (.node (.input ⟨62, by decide⟩) (.input ⟨63, by decide⟩)))

def cross64 : OrderedTree 64 :=
  ⟨.node (.node (.node (cross64_0) (cross64_4)) (.node (cross64_2) (cross64_6))) (.node (.node (cross64_1) (cross64_5)) (.node (cross64_3) (cross64_7))), by decide⟩

def local128_0 : Tree 128 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)) (.input ⟨8, by decide⟩)) (.input ⟨9, by decide⟩)) (.input ⟨16, by decide⟩)) (.input ⟨17, by decide⟩)) (.input ⟨24, by decide⟩)) (.input ⟨25, by decide⟩)) (.input ⟨32, by decide⟩)) (.input ⟨33, by decide⟩)) (.input ⟨40, by decide⟩)) (.input ⟨41, by decide⟩)) (.input ⟨48, by decide⟩)) (.input ⟨49, by decide⟩)) (.input ⟨56, by decide⟩)) (.input ⟨57, by decide⟩)) (.input ⟨64, by decide⟩)) (.input ⟨65, by decide⟩)) (.input ⟨72, by decide⟩)) (.input ⟨73, by decide⟩)) (.input ⟨80, by decide⟩)) (.input ⟨81, by decide⟩)) (.input ⟨88, by decide⟩)) (.input ⟨89, by decide⟩)) (.input ⟨96, by decide⟩)) (.input ⟨97, by decide⟩)) (.input ⟨104, by decide⟩)) (.input ⟨105, by decide⟩)) (.input ⟨112, by decide⟩)) (.input ⟨113, by decide⟩)) (.input ⟨120, by decide⟩)) (.input ⟨121, by decide⟩)

def local128_1 : Tree 128 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨2, by decide⟩) (.input ⟨3, by decide⟩)) (.input ⟨10, by decide⟩)) (.input ⟨11, by decide⟩)) (.input ⟨18, by decide⟩)) (.input ⟨19, by decide⟩)) (.input ⟨26, by decide⟩)) (.input ⟨27, by decide⟩)) (.input ⟨34, by decide⟩)) (.input ⟨35, by decide⟩)) (.input ⟨42, by decide⟩)) (.input ⟨43, by decide⟩)) (.input ⟨50, by decide⟩)) (.input ⟨51, by decide⟩)) (.input ⟨58, by decide⟩)) (.input ⟨59, by decide⟩)) (.input ⟨66, by decide⟩)) (.input ⟨67, by decide⟩)) (.input ⟨74, by decide⟩)) (.input ⟨75, by decide⟩)) (.input ⟨82, by decide⟩)) (.input ⟨83, by decide⟩)) (.input ⟨90, by decide⟩)) (.input ⟨91, by decide⟩)) (.input ⟨98, by decide⟩)) (.input ⟨99, by decide⟩)) (.input ⟨106, by decide⟩)) (.input ⟨107, by decide⟩)) (.input ⟨114, by decide⟩)) (.input ⟨115, by decide⟩)) (.input ⟨122, by decide⟩)) (.input ⟨123, by decide⟩)

def local128_2 : Tree 128 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨4, by decide⟩) (.input ⟨5, by decide⟩)) (.input ⟨12, by decide⟩)) (.input ⟨13, by decide⟩)) (.input ⟨20, by decide⟩)) (.input ⟨21, by decide⟩)) (.input ⟨28, by decide⟩)) (.input ⟨29, by decide⟩)) (.input ⟨36, by decide⟩)) (.input ⟨37, by decide⟩)) (.input ⟨44, by decide⟩)) (.input ⟨45, by decide⟩)) (.input ⟨52, by decide⟩)) (.input ⟨53, by decide⟩)) (.input ⟨60, by decide⟩)) (.input ⟨61, by decide⟩)) (.input ⟨68, by decide⟩)) (.input ⟨69, by decide⟩)) (.input ⟨76, by decide⟩)) (.input ⟨77, by decide⟩)) (.input ⟨84, by decide⟩)) (.input ⟨85, by decide⟩)) (.input ⟨92, by decide⟩)) (.input ⟨93, by decide⟩)) (.input ⟨100, by decide⟩)) (.input ⟨101, by decide⟩)) (.input ⟨108, by decide⟩)) (.input ⟨109, by decide⟩)) (.input ⟨116, by decide⟩)) (.input ⟨117, by decide⟩)) (.input ⟨124, by decide⟩)) (.input ⟨125, by decide⟩)

def local128_3 : Tree 128 :=
  .node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.input ⟨6, by decide⟩) (.input ⟨7, by decide⟩)) (.input ⟨14, by decide⟩)) (.input ⟨15, by decide⟩)) (.input ⟨22, by decide⟩)) (.input ⟨23, by decide⟩)) (.input ⟨30, by decide⟩)) (.input ⟨31, by decide⟩)) (.input ⟨38, by decide⟩)) (.input ⟨39, by decide⟩)) (.input ⟨46, by decide⟩)) (.input ⟨47, by decide⟩)) (.input ⟨54, by decide⟩)) (.input ⟨55, by decide⟩)) (.input ⟨62, by decide⟩)) (.input ⟨63, by decide⟩)) (.input ⟨70, by decide⟩)) (.input ⟨71, by decide⟩)) (.input ⟨78, by decide⟩)) (.input ⟨79, by decide⟩)) (.input ⟨86, by decide⟩)) (.input ⟨87, by decide⟩)) (.input ⟨94, by decide⟩)) (.input ⟨95, by decide⟩)) (.input ⟨102, by decide⟩)) (.input ⟨103, by decide⟩)) (.input ⟨110, by decide⟩)) (.input ⟨111, by decide⟩)) (.input ⟨118, by decide⟩)) (.input ⟨119, by decide⟩)) (.input ⟨126, by decide⟩)) (.input ⟨127, by decide⟩)

def tree128_0 : Tree 128 := .node (.node (local128_0) (local128_2)) (.node (local128_1) (local128_3))
theorem complete128_0 : tree128_0.leaves.Perm (List.finRange 128) := by decide

def tree128_1 : Tree 128 := .node (.node (local128_1) (local128_3)) (.node (local128_0) (local128_2))
theorem complete128_1 : tree128_1.leaves.Perm (List.finRange 128) := by decide

def tree128_2 : Tree 128 := .node (.node (local128_2) (local128_0)) (.node (local128_3) (local128_1))
theorem complete128_2 : tree128_2.leaves.Perm (List.finRange 128) := by decide

def tree128_3 : Tree 128 := .node (.node (local128_3) (local128_1)) (.node (local128_2) (local128_0))
theorem complete128_3 : tree128_3.leaves.Perm (List.finRange 128) := by decide

def warp128 (column : Nat) : OrderedTree 128 :=
  if column%8/2=0 then ⟨tree128_0,complete128_0⟩ else
  if column%8/2=1 then ⟨tree128_1,complete128_1⟩ else
  if column%8/2=2 then ⟨tree128_2,complete128_2⟩ else
  ⟨tree128_3,complete128_3⟩

def cross128_0 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)) (.input ⟨64, by decide⟩)) (.input ⟨65, by decide⟩)) (.node (.node (.node (.input ⟨4, by decide⟩) (.input ⟨5, by decide⟩)) (.input ⟨68, by decide⟩)) (.input ⟨69, by decide⟩))) (.node (.node (.node (.node (.input ⟨2, by decide⟩) (.input ⟨3, by decide⟩)) (.input ⟨66, by decide⟩)) (.input ⟨67, by decide⟩)) (.node (.node (.node (.input ⟨6, by decide⟩) (.input ⟨7, by decide⟩)) (.input ⟨70, by decide⟩)) (.input ⟨71, by decide⟩)))

def cross128_1 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨8, by decide⟩) (.input ⟨9, by decide⟩)) (.input ⟨72, by decide⟩)) (.input ⟨73, by decide⟩)) (.node (.node (.node (.input ⟨12, by decide⟩) (.input ⟨13, by decide⟩)) (.input ⟨76, by decide⟩)) (.input ⟨77, by decide⟩))) (.node (.node (.node (.node (.input ⟨10, by decide⟩) (.input ⟨11, by decide⟩)) (.input ⟨74, by decide⟩)) (.input ⟨75, by decide⟩)) (.node (.node (.node (.input ⟨14, by decide⟩) (.input ⟨15, by decide⟩)) (.input ⟨78, by decide⟩)) (.input ⟨79, by decide⟩)))

def cross128_2 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨16, by decide⟩) (.input ⟨17, by decide⟩)) (.input ⟨80, by decide⟩)) (.input ⟨81, by decide⟩)) (.node (.node (.node (.input ⟨20, by decide⟩) (.input ⟨21, by decide⟩)) (.input ⟨84, by decide⟩)) (.input ⟨85, by decide⟩))) (.node (.node (.node (.node (.input ⟨18, by decide⟩) (.input ⟨19, by decide⟩)) (.input ⟨82, by decide⟩)) (.input ⟨83, by decide⟩)) (.node (.node (.node (.input ⟨22, by decide⟩) (.input ⟨23, by decide⟩)) (.input ⟨86, by decide⟩)) (.input ⟨87, by decide⟩)))

def cross128_3 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨24, by decide⟩) (.input ⟨25, by decide⟩)) (.input ⟨88, by decide⟩)) (.input ⟨89, by decide⟩)) (.node (.node (.node (.input ⟨28, by decide⟩) (.input ⟨29, by decide⟩)) (.input ⟨92, by decide⟩)) (.input ⟨93, by decide⟩))) (.node (.node (.node (.node (.input ⟨26, by decide⟩) (.input ⟨27, by decide⟩)) (.input ⟨90, by decide⟩)) (.input ⟨91, by decide⟩)) (.node (.node (.node (.input ⟨30, by decide⟩) (.input ⟨31, by decide⟩)) (.input ⟨94, by decide⟩)) (.input ⟨95, by decide⟩)))

def cross128_4 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨32, by decide⟩) (.input ⟨33, by decide⟩)) (.input ⟨96, by decide⟩)) (.input ⟨97, by decide⟩)) (.node (.node (.node (.input ⟨36, by decide⟩) (.input ⟨37, by decide⟩)) (.input ⟨100, by decide⟩)) (.input ⟨101, by decide⟩))) (.node (.node (.node (.node (.input ⟨34, by decide⟩) (.input ⟨35, by decide⟩)) (.input ⟨98, by decide⟩)) (.input ⟨99, by decide⟩)) (.node (.node (.node (.input ⟨38, by decide⟩) (.input ⟨39, by decide⟩)) (.input ⟨102, by decide⟩)) (.input ⟨103, by decide⟩)))

def cross128_5 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨40, by decide⟩) (.input ⟨41, by decide⟩)) (.input ⟨104, by decide⟩)) (.input ⟨105, by decide⟩)) (.node (.node (.node (.input ⟨44, by decide⟩) (.input ⟨45, by decide⟩)) (.input ⟨108, by decide⟩)) (.input ⟨109, by decide⟩))) (.node (.node (.node (.node (.input ⟨42, by decide⟩) (.input ⟨43, by decide⟩)) (.input ⟨106, by decide⟩)) (.input ⟨107, by decide⟩)) (.node (.node (.node (.input ⟨46, by decide⟩) (.input ⟨47, by decide⟩)) (.input ⟨110, by decide⟩)) (.input ⟨111, by decide⟩)))

def cross128_6 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨48, by decide⟩) (.input ⟨49, by decide⟩)) (.input ⟨112, by decide⟩)) (.input ⟨113, by decide⟩)) (.node (.node (.node (.input ⟨52, by decide⟩) (.input ⟨53, by decide⟩)) (.input ⟨116, by decide⟩)) (.input ⟨117, by decide⟩))) (.node (.node (.node (.node (.input ⟨50, by decide⟩) (.input ⟨51, by decide⟩)) (.input ⟨114, by decide⟩)) (.input ⟨115, by decide⟩)) (.node (.node (.node (.input ⟨54, by decide⟩) (.input ⟨55, by decide⟩)) (.input ⟨118, by decide⟩)) (.input ⟨119, by decide⟩)))

def cross128_7 : Tree 128 :=
  .node (.node (.node (.node (.node (.input ⟨56, by decide⟩) (.input ⟨57, by decide⟩)) (.input ⟨120, by decide⟩)) (.input ⟨121, by decide⟩)) (.node (.node (.node (.input ⟨60, by decide⟩) (.input ⟨61, by decide⟩)) (.input ⟨124, by decide⟩)) (.input ⟨125, by decide⟩))) (.node (.node (.node (.node (.input ⟨58, by decide⟩) (.input ⟨59, by decide⟩)) (.input ⟨122, by decide⟩)) (.input ⟨123, by decide⟩)) (.node (.node (.node (.input ⟨62, by decide⟩) (.input ⟨63, by decide⟩)) (.input ⟨126, by decide⟩)) (.input ⟨127, by decide⟩)))

def cross128 : OrderedTree 128 :=
  ⟨.node (.node (.node (cross128_0) (cross128_4)) (.node (cross128_2) (cross128_6))) (.node (.node (cross128_1) (cross128_5)) (.node (cross128_3) (cross128_7))), by decide⟩

def foldTree (head : Native.Tree n) (tail : List (Fin n)) : Native.Tree n :=
  tail.foldl (fun t i => .node t (.input i)) head

theorem foldTree_leaves (head : Native.Tree n) (tail : List (Fin n)) :
    (foldTree head tail).leaves=head.leaves++tail := by
  induction tail generalizing head with
  | nil => simp [foldTree]
  | cons i rest ih =>
    simpa [foldTree,List.foldl_cons,Native.Tree.leaves,List.append_assoc] using ih (.node head (.input i))

/-- Total interface fallback; admitted source reductions only use 64 or 128. -/
def fallback (n : Nat) (positive : 0<n) : OrderedTree n := by
  cases n with
  | zero => omega
  | succ n =>
    exact ⟨foldTree (.input 0) ((List.finRange n).map Fin.succ), by
      rw [foldTree_leaves]
      simp [Native.Tree.leaves,List.finRange_succ]⟩

def reductionTree (crossWarp : Bool) (n : Nat) (positive : 0<n) (column : Nat) : OrderedTree n :=
  if h : n=64 then h ▸ (if crossWarp then cross64 else warp64 column)
  else if h : n=128 then h ▸ (if crossWarp then cross128 else warp128 column)
  else fallback n positive

theorem reduction_complete (crossWarp : Bool) (n : Nat) (positive : 0<n) (column : Nat) :
    (reductionTree crossWarp n positive column).tree.leaves.Perm (List.finRange n) :=
  (reductionTree crossWarp n positive column).complete

theorem replicas_differ : tree64_0.leaves ≠ tree64_1.leaves := by decide

#axiomsClean reduction_complete
#axiomsClean replicas_differ
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelNative

import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Raw

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.NativeSchedule
open VeriTile.Triton Host Raw
noncomputable section

def local16_0 : Tree 16 := (.node (.node (.node (.leaf ⟨0,by decide⟩) (.leaf ⟨1,by decide⟩)) (.leaf ⟨8,by decide⟩)) (.leaf ⟨9,by decide⟩))

def local16_1 : Tree 16 := (.node (.node (.node (.leaf ⟨2,by decide⟩) (.leaf ⟨3,by decide⟩)) (.leaf ⟨10,by decide⟩)) (.leaf ⟨11,by decide⟩))

def local16_2 : Tree 16 := (.node (.node (.node (.leaf ⟨4,by decide⟩) (.leaf ⟨5,by decide⟩)) (.leaf ⟨12,by decide⟩)) (.leaf ⟨13,by decide⟩))

def local16_3 : Tree 16 := (.node (.node (.node (.leaf ⟨6,by decide⟩) (.leaf ⟨7,by decide⟩)) (.leaf ⟨14,by decide⟩)) (.leaf ⟨15,by decide⟩))

def tree16 (owner : Nat) : Tree 16 :=
  if owner%4=0 then .node (.node local16_0 local16_2) (.node local16_1 local16_3)
  else if owner%4=1 then .node (.node local16_1 local16_3) (.node local16_0 local16_2)
  else if owner%4=2 then .node (.node local16_2 local16_0) (.node local16_3 local16_1)
  else .node (.node local16_3 local16_1) (.node local16_2 local16_0)

theorem complete16 (owner : Nat) : (tree16 owner).indices.Perm (List.finRange 16) := by
  unfold tree16
  split
  · decide
  · split
    · decide
    · split <;> decide

def order16 (owner : Nat) : Order 16 := ⟨tree16 owner,complete16 owner⟩

def local32_0 : Tree 32 := (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨0,by decide⟩) (.leaf ⟨1,by decide⟩)) (.leaf ⟨8,by decide⟩)) (.leaf ⟨9,by decide⟩)) (.leaf ⟨16,by decide⟩)) (.leaf ⟨17,by decide⟩)) (.leaf ⟨24,by decide⟩)) (.leaf ⟨25,by decide⟩))

def local32_1 : Tree 32 := (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨2,by decide⟩) (.leaf ⟨3,by decide⟩)) (.leaf ⟨10,by decide⟩)) (.leaf ⟨11,by decide⟩)) (.leaf ⟨18,by decide⟩)) (.leaf ⟨19,by decide⟩)) (.leaf ⟨26,by decide⟩)) (.leaf ⟨27,by decide⟩))

def local32_2 : Tree 32 := (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨4,by decide⟩) (.leaf ⟨5,by decide⟩)) (.leaf ⟨12,by decide⟩)) (.leaf ⟨13,by decide⟩)) (.leaf ⟨20,by decide⟩)) (.leaf ⟨21,by decide⟩)) (.leaf ⟨28,by decide⟩)) (.leaf ⟨29,by decide⟩))

def local32_3 : Tree 32 := (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨6,by decide⟩) (.leaf ⟨7,by decide⟩)) (.leaf ⟨14,by decide⟩)) (.leaf ⟨15,by decide⟩)) (.leaf ⟨22,by decide⟩)) (.leaf ⟨23,by decide⟩)) (.leaf ⟨30,by decide⟩)) (.leaf ⟨31,by decide⟩))

def tree32 (owner : Nat) : Tree 32 :=
  if owner%4=0 then .node (.node local32_0 local32_2) (.node local32_1 local32_3)
  else if owner%4=1 then .node (.node local32_1 local32_3) (.node local32_0 local32_2)
  else if owner%4=2 then .node (.node local32_2 local32_0) (.node local32_3 local32_1)
  else .node (.node local32_3 local32_1) (.node local32_2 local32_0)

theorem complete32 (owner : Nat) : (tree32 owner).indices.Perm (List.finRange 32) := by
  unfold tree32
  split
  · decide
  · split
    · decide
    · split <;> decide

def order32 (owner : Nat) : Order 32 := ⟨tree32 owner,complete32 owner⟩

def local64_0 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨0,by decide⟩) (.leaf ⟨1,by decide⟩)) (.leaf ⟨8,by decide⟩)) (.leaf ⟨9,by decide⟩)) (.leaf ⟨16,by decide⟩)) (.leaf ⟨17,by decide⟩)) (.leaf ⟨24,by decide⟩)) (.leaf ⟨25,by decide⟩)) (.leaf ⟨32,by decide⟩)) (.leaf ⟨33,by decide⟩)) (.leaf ⟨40,by decide⟩)) (.leaf ⟨41,by decide⟩)) (.leaf ⟨48,by decide⟩)) (.leaf ⟨49,by decide⟩)) (.leaf ⟨56,by decide⟩)) (.leaf ⟨57,by decide⟩))

def local64_1 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨2,by decide⟩) (.leaf ⟨3,by decide⟩)) (.leaf ⟨10,by decide⟩)) (.leaf ⟨11,by decide⟩)) (.leaf ⟨18,by decide⟩)) (.leaf ⟨19,by decide⟩)) (.leaf ⟨26,by decide⟩)) (.leaf ⟨27,by decide⟩)) (.leaf ⟨34,by decide⟩)) (.leaf ⟨35,by decide⟩)) (.leaf ⟨42,by decide⟩)) (.leaf ⟨43,by decide⟩)) (.leaf ⟨50,by decide⟩)) (.leaf ⟨51,by decide⟩)) (.leaf ⟨58,by decide⟩)) (.leaf ⟨59,by decide⟩))

def local64_2 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨4,by decide⟩) (.leaf ⟨5,by decide⟩)) (.leaf ⟨12,by decide⟩)) (.leaf ⟨13,by decide⟩)) (.leaf ⟨20,by decide⟩)) (.leaf ⟨21,by decide⟩)) (.leaf ⟨28,by decide⟩)) (.leaf ⟨29,by decide⟩)) (.leaf ⟨36,by decide⟩)) (.leaf ⟨37,by decide⟩)) (.leaf ⟨44,by decide⟩)) (.leaf ⟨45,by decide⟩)) (.leaf ⟨52,by decide⟩)) (.leaf ⟨53,by decide⟩)) (.leaf ⟨60,by decide⟩)) (.leaf ⟨61,by decide⟩))

def local64_3 : Tree 64 := (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.node (.leaf ⟨6,by decide⟩) (.leaf ⟨7,by decide⟩)) (.leaf ⟨14,by decide⟩)) (.leaf ⟨15,by decide⟩)) (.leaf ⟨22,by decide⟩)) (.leaf ⟨23,by decide⟩)) (.leaf ⟨30,by decide⟩)) (.leaf ⟨31,by decide⟩)) (.leaf ⟨38,by decide⟩)) (.leaf ⟨39,by decide⟩)) (.leaf ⟨46,by decide⟩)) (.leaf ⟨47,by decide⟩)) (.leaf ⟨54,by decide⟩)) (.leaf ⟨55,by decide⟩)) (.leaf ⟨62,by decide⟩)) (.leaf ⟨63,by decide⟩))

def tree64 (owner : Nat) : Tree 64 :=
  if owner%4=0 then .node (.node local64_0 local64_2) (.node local64_1 local64_3)
  else if owner%4=1 then .node (.node local64_1 local64_3) (.node local64_0 local64_2)
  else if owner%4=2 then .node (.node local64_2 local64_0) (.node local64_3 local64_1)
  else .node (.node local64_3 local64_1) (.node local64_2 local64_0)

theorem complete64 (owner : Nat) : (tree64 owner).indices.Perm (List.finRange 64) := by
  unfold tree64
  split
  · decide
  · split
    · decide
    · split <;> decide

def order64 (owner : Nat) : Order 64 := ⟨tree64 owner,complete64 owner⟩

/-- Exactly the six compiled and independently replayed profiles. This is
not a theorem about unexamined compiler versions or profile combinations. -/
inductive Kind where
  | b64d16 | b32d16 | b64d32 | b64d64 | b64d128 | b16d16
  deriving DecidableEq

def Kind.block : Kind → Nat | .b32d16=>32 | .b16d16=>16 | _=>64
def Kind.dim : Kind → Nat | .b64d32=>32 | .b64d64=>64 | .b64d128=>128 | _=>16
def Kind.init : Kind → Nat | .b32d16 | .b64d64=>0 | _=>1
def Kind.finish : Kind → Nat | .b64d16 | .b64d64 | .b64d128=>1 | _=>0
def Kind.window : Kind → Nat | .b64d32 | .b64d128=>0 | _=>1
def Kind.complement : Kind → Nat | .b64d64=>1 | _=>0
def Kind.even : Kind → Nat | .b64d32 | .b64d128=>1 | _=>0

def Kind.order (k : Kind) (owner : Nat) : Order k.block :=
  match k with
  | .b32d16=>order32 owner
  | .b16d16=>order16 owner
  | .b64d16 | .b64d32 | .b64d64 | .b64d128=>order64 owner

def Kind.qk : Kind → Matrix | .b32d16 | .b16d16=>.mma16n8 | _=>.wgmma64n64
def Kind.pv : Kind → Matrix
  | .b32d16 | .b16d16=>.mma16n8
  | .b64d16=>.wgmma64n16 | .b64d32=>.wgmma64n32
  | .b64d64=>.wgmma64n64 | .b64d128=>.wgmma64n128

def Kind.fusedScore : Kind → Bool | .b64d32 | .b64d128=>Bool.true | _=>Bool.false

structure Profile (c : Call) (k : Kind) : Prop where
  bm : c.BM=k.block
  bn : c.BN=k.block
  d : c.D=k.dim
  init : c.init=k.init
  finish : c.finish=k.finish
  window : c.window=k.window
  complement : c.complement=k.complement
  evenM : c.params.evenM=k.even
  evenN : c.params.evenN=k.even

def qkOwner (row col : Nat) := row%64/16*32+row%8*4+col%8/2

/-- Finite physical choices only. BM64 selects one of four same-warp
writers. BM32 has two replicated warps: each warp selects one writer,
then each read byte selects one of those two candidates. BM16 uses a
checked shuffle and has no scalar shared-store choice. M and L choices
are independent because their stores are different instructions. -/
structure Choice where
  lane : Fin 2 → Fin 4
  replica : Fin 4 → Fin 2

structure Selection (c : Call) where
  m : GridIndex c.grid → Fin c.BM → Choice
  l : GridIndex c.grid → Fin c.BM → Choice

def selected (k : Kind) (choice : Choice) (byte : Fin 4) : Fin 4 :=
  match k with
  | .b16d16=>0
  | .b32d16=>choice.lane (choice.replica byte)
  | _=>choice.lane 0

def schedule (c : Call) (k : Kind) (h : Profile c k) (winner : Selection c) : Schedule c where
  qk := k.qk
  pv := k.pv
  fuseScoreSubtract := k.fusedScore
  maxOrder := fun _ owner=>h.bn.symm ▸ k.order owner
  sumOrder := fun _ owner=>h.bn.symm ▸ k.order owner
  probabilityOwner := fun r j=>qkOwner r.val j.val
  pvProbabilityOwner := fun r j _=>qkOwner r.val j.val
  accumulatorAlphaOwner := fun _ _ owner=>owner
  centerStoreOwner := fun idx r byte=>qkOwner r.val 0+(selected k (winner.m idx r) byte).val
  denominatorStoreOwner := fun idx r byte=>qkOwner r.val 0+(selected k (winner.l idx r) byte).val
  outputStoreOwner := fun r d=>qkOwner r.val d.val
  normalizationOwner := fun _ _ owner=>owner

theorem selected_owner (c : Call) (k : Kind) (h : Profile c k) (winner : Selection c)
    (idx : GridIndex c.grid) (r : Fin c.BM) (byte : Fin 4) :
    (schedule c k h winner).centerStoreOwner idx r byte ∈
      (List.finRange 4).map (fun lane=>qkOwner r.val 0+lane.val) := by
  simp [schedule]

theorem direct16 (c : Call) (h : Profile c .b16d16) (winner : Selection c)
    (idx : GridIndex c.grid) (r : Fin c.BM) (byte : Fin 4) :
    (schedule c .b16d16 h winner).centerStoreOwner idx r byte=qkOwner r.val 0 := by
  simp [schedule,selected]

#print axioms complete16
#print axioms complete32
#print axioms complete64
#print axioms selected_owner
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.NativeSchedule

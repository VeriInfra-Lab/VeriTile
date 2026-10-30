import bench.tritonbench_g.softmax_reducev.Proofs.NativeGraph
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.NativeGraph
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 12000000
set_option maxRecDepth 65536

/-- SM80 structural layout parameters captured from actual JIT TTGIR/PTX.
Shared-store replica choice is retained per output channel; different contenders are
not silently equated by an assumed floating commutativity law. -/
structure Profile where
  qVector2 : Bool
  valueRowLog : Fin 6
  valueVector : Nat
  reverseSumRoot : Bool
  storeRow : Nat → Nat

def qVector (v : Bool) : Nat := if v then 2 else 1

def key64 (k : Nat) : Fin 64 := ⟨k%64,Nat.mod_lt _ (by decide)⟩
def lane32 (t : Nat) : Fin 32 := ⟨t%32,Nat.mod_lt _ (by decide)⟩

def inputTree (qv : Bool) (k : Nat) : Tree 64 :=
  .input (key64 k) ((k%64/qVector qv)%32)

def foldTree (fusion : Fusion) (padding : Word) (xs : List (Tree n)) : Tree n :=
  match xs with
  | [] => .padding padding
  | x::xs => xs.foldl (.join fusion) x

def qLocal (v : Bool) (t : Nat) : Tree 64 :=
  if v then .join .separate (inputTree v (2*(t%32))) (inputTree v (2*(t%32)+1))
  else .join .separate (inputTree v (t%32)) (inputTree v (t%32+32))

def butterfly (reverseRoot : Bool) (distances : List Nat) (initial : Nat → Tree n) : Nat → Tree n :=
  distances.foldl (fun previous distance => fun t =>
    if reverseRoot && decide (distance=1) then .join .separate (previous (Nat.xor t distance)) (previous t)
    else .join .separate (previous t) (previous (Nat.xor t distance))) initial

def qTree (v reverseRoot : Bool) (t : Nat) : Tree 64 :=
  butterfly reverseRoot [16,8,4,2,1] (qLocal v) t

def valueLocal (rowLog : Fin 6) (qv : Bool) (t : Nat) : Tree 64 :=
  let rows := 2^rowLog.val
  let dimThreads := 32/rows
  foldTree .both 0 ((List.range (64/rows)).map
    (fun slot => inputTree qv ((t%32)/dimThreads+slot*rows)))

def valueTree (rowLog : Fin 6) (qv : Bool) (t : Nat) : Tree 64 :=
  butterfly Bool.false ([16,8,4,2,1].filter (fun d => decide (32/(2^rowLog.val) ≤ d)))
    (valueLocal rowLog qv) t

theorem q_complete : ∀ v reverseRoot : Bool, ∀ t : Fin 32,
    (qTree v reverseRoot t.val).indices.Perm (List.finRange 64) := by decide

theorem value_complete : ∀ rowLog : Fin 6, ∀ qv : Bool, ∀ t : Fin 32,
    (valueTree rowLog qv t.val).indices.Perm (List.finRange 64) := by decide

def orderedQ (v reverseRoot : Bool) (t : Nat) : OrderedTree 64 :=
  ⟨qTree v reverseRoot (lane32 t).val,q_complete v reverseRoot (lane32 t)⟩

def orderedV (rowLog : Fin 6) (qv : Bool) (t : Nat) : OrderedTree 64 :=
  ⟨valueTree rowLog qv (lane32 t).val,value_complete rowLog qv (lane32 t)⟩

def fallback (n : Nat) : Tree n :=
  (List.finRange n).foldl (fun previous i => .join .separate previous (.input i 0)) (.padding 0)

theorem fold_indices (xs : List (Fin n)) (initial : Tree n) :
    (xs.foldl (fun previous i => Tree.join .separate previous (.input i 0)) initial).indices=
      initial.indices++xs := by
  induction xs generalizing initial with
  | nil => simp
  | cons x xs ih => simp [List.foldl_cons,ih,Tree.indices,List.append_assoc]

def fallbackOrdered (n : Nat) : OrderedTree n :=
  ⟨fallback n,by simp [fallback,fold_indices,Tree.indices]⟩

def adapt (n : Nat) (actual : OrderedTree 64) : OrderedTree n :=
  if h : n=64 then h.symm ▸ actual else fallbackOrdered n

/-- Concrete, nonvacuous profile: the only reached reductions have axis 64.
The source rank distinguishes the Q/max/denominator trees from the gathered
V-product tree. Every local product add retains actual FMA selection. -/
def profilePlan (p : Profile) : Plan where
  sumTree := fun {shape} ax _ _ t => adapt (TileShape.axisDim shape ax)
    (if shape.length=1 then orderedQ p.qVector2 p.reverseSumRoot t
     else orderedV p.valueRowLog p.qVector2 t)
  maxTree := fun {_} ax _ _ t => adapt (TileShape.axisDim _ ax) (orderedQ p.qVector2 Bool.false t)
  storeReplica := fun _ d => (d/p.valueVector)%(32/(2^p.valueRowLog.val))+p.storeRow d*(32/(2^p.valueRowLog.val))

/-- The saved profiles use these exact layout choices; unaligned D16 changes
V's vector width and its row reduction, while D128 reverses its last sum add. -/
def sm80Profile (dimensions : Nat) (unaligned : Bool) (sharedRow : Nat → Nat) : Profile :=
  if unaligned then ⟨Bool.false,⟨1,by decide⟩,1,Bool.false,(fun _ => 0)⟩
  else match dimensions with
    | 1 => ⟨Bool.false,⟨5,by decide⟩,1,Bool.false,(fun _ => 0)⟩
    | 2 => ⟨Bool.false,⟨4,by decide⟩,1,Bool.false,(fun _ => 0)⟩
    | 4 => ⟨Bool.false,⟨3,by decide⟩,1,Bool.false,(fun _ => 0)⟩
    | 8 => ⟨Bool.false,⟨2,by decide⟩,1,Bool.false,(fun _ => 0)⟩
    | 16 => ⟨Bool.false,⟨3,by decide⟩,4,Bool.false,(fun d => sharedRow d%8)⟩
    | 32 => ⟨Bool.false,⟨2,by decide⟩,4,Bool.false,(fun d => sharedRow d%4)⟩
    | 64 => ⟨Bool.true,⟨1,by decide⟩,4,Bool.false,(fun _ => 0)⟩
    | _ => ⟨Bool.false,⟨0,by decide⟩,4,Bool.true,(fun _ => 0)⟩

#axiomsClean q_complete
#axiomsClean value_complete
#axiomsClean fold_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.NativeGraph

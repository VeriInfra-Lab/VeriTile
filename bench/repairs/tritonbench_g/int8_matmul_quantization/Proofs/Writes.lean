import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Events

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000

abbrev Cell (F H : Type) := Sum Byte (Stored F H)
def put (m : Memory F H) (p : Address) (value : Cell F H) : Memory F H :=
  fun q => if q=p then value else m q
def writeAll (address : α → Address) (value : α → Cell F H) : List α → Memory F H → Memory F H
  | [],m => m
  | x::xs,m => writeAll address value xs (put m (address x) (value x))

theorem writeAll_append (address : α → Address) (value : α → Cell F H)
    (xs ys : List α) (m : Memory F H) :
    writeAll address value (xs++ys) m=writeAll address value ys (writeAll address value xs m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xs ih => exact ih _
theorem writeAll_map (address : α → Address) (value : α → Cell F H) (f : β → α)
    (xs : List β) (m : Memory F H) :
    writeAll address value (xs.map f) m=writeAll (fun x => address (f x)) (fun x => value (f x)) xs m := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xs ih => exact ih _

theorem writeAll_frame (address : α → Address) (value : α → Cell F H) (xs : List α)
    (m : Memory F H) (p : Address) (outside : ∀ x,x∈xs → p≠address x) :
    writeAll address value xs m p=m p := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xs ih =>
    rw [writeAll,ih _ (fun y hy => outside y (List.mem_cons_of_mem _ hy))]
    exact if_neg (outside x (by simp))

theorem writeAll_at [DecidableEq α] (address : α → Address) (value : α → Cell F H)
    (injective : Function.Injective address) (xs : List α) (m : Memory F H) (x : α) (present : x∈xs) :
    writeAll address value xs m (address x)=value x := by
  induction xs generalizing m with
  | nil => simp at present
  | cons y xs ih =>
    rw [writeAll]
    by_cases inside : x∈xs
    · exact ih _ inside
    · have eq : x=y := by simpa [inside] using present
      rw [writeAll_frame address value xs _ (address x)]
      · simp [put,eq]
      · intro z hz same
        exact inside ((injective same) ▸ hz)

def putCode (p : Address) : Cell F H → Code I F H Unit
  | .inl x => emit (.store8 p x)
  | .inr (.inl x) => emit (.store32 p x)
  | .inr (.inr x) => emit (.store p x)
theorem eval_putCode (a : Ops I F H) (p : Address) (value : Cell F H) (m : Memory F H) :
    (putCode p value).eval a m=((),put m p value) := by
  cases value with
  | inl x => rfl
  | inr x => cases x <;> rfl
def writeAllCode (address : α → Address) (value : α → Cell F H) : List α → Code I F H Unit
  | [] => pure ()
  | x::xs => do
      let _ ← putCode (address x) (value x)
      writeAllCode address value xs
theorem eval_writeAllCode (a : Ops I F H) (address : α → Address) (value : α → Cell F H)
    (xs : List α) (m : Memory F H) :
    (writeAllCode address value xs).eval a m=((),writeAll address value xs m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xs ih => simp only [writeAllCode,bind,eval_bind,eval_putCode,Prod.fst,Prod.snd,ih,writeAll]

def sequenceCode (f : α → Code I F H Unit) : List α → Code I F H Unit
  | [] => pure ()
  | x::xs => do
      let _ ← f x
      sequenceCode f xs

#axiomsClean writeAll_frame
#axiomsClean writeAll_at
#axiomsClean eval_putCode
#axiomsClean eval_writeAllCode
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization

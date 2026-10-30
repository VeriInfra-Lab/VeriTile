import VeriTile.Triton.Concurrent.EventIR
import VeriTile.Meta.StatementAudit

/-! Safety is monotone in allocation capacity. Enlarging a region never
excuses a negative access, and all expression reads remain checked. -/
namespace VeriTile.Triton.Concurrent

structure Bounds.LE (small large : Bounds) : Prop where
  input : ∀ region,small.input region≤large.input region
  mem : ∀ region,small.mem region≤large.mem region
  words : ∀ region,small.words region≤large.words region

theorem InWindow.mono {small large : Nat} {address : Int}
    (capacity : small≤large) (safe : InWindow small address) : InWindow large address :=
  ⟨safe.1,lt_of_lt_of_le safe.2 (Int.ofNat_le.mpr capacity)⟩

theorem Expr.Safe.mono (small large : Bounds) (capacity : small.LE large)
    (expr : Expr) (safe : expr.Safe small) : expr.Safe large := by
  induction expr with
  | zero => trivial
  | input channel index => exact InWindow.mono (capacity.input channel) safe
  | load region address => exact InWindow.mono (capacity.mem region) safe
  | register thread lane => trivial
  | add left right ihl ihr => exact ⟨ihl safe.1,ihr safe.2⟩
  | mma8 left right seed ihl ihr ihs =>
    exact ⟨fun i=>ihl i (safe.1 i),fun i=>ihr i (safe.2.1 i),ihs safe.2.2⟩
  | halfRoundTrip value ih => exact ih safe

theorem Event.Safe.mono (small large : Bounds) (capacity : small.LE large)
    (event : Event) (safe : event.Safe small) : event.Safe large := by
  cases event with
  | store region address value | atomicAdd region address value =>
    exact ⟨InWindow.mono (capacity.mem region) safe.1,Expr.Safe.mono small large capacity value safe.2⟩
  | assign thread lane value | readOnly value =>
    exact Expr.Safe.mono small large capacity value safe
  | compareExchange region address compare replace thread destination | exchange region address word =>
    exact InWindow.mono (capacity.words region) safe
  | barrier | silent => trivial

theorem TapeSafe.mono (small large : Bounds) (capacity : small.LE large)
    (events : List Event) (safe : TapeSafe small events) : TapeSafe large events :=
  fun event member=>Event.Safe.mono small large capacity event (safe event member)

#axiomsClean Expr.Safe.mono
#axiomsClean Event.Safe.mono
#axiomsClean TapeSafe.mono
end VeriTile.Triton.Concurrent

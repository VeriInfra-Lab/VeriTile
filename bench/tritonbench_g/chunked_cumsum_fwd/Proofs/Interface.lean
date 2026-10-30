import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.IO

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar Kernel.Math
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem initial_eq {α : Type} (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) (l : Launch) (s : State (Term α)) :
    (invocation M c plan hasBias useSoftplus lower upper l).initial s =
      tileState l (bindBounds s (.input lower) (.input upper)) := by rfl

theorem schedule_eq {α : Type} (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) (ls : List Launch) (s : State (Term α)) :
    runSchedule (ls.map (invocation M c plan hasBias useSoftplus lower upper)) s =
      runTiles (graphAlgebra M c plan) c hasBias useSoftplus (.input lower) (.input upper) ls s := by
  induction ls generalizing s with
  | nil => rfl
  | cons l ls ih =>
    simp only [List.map_cons, runSchedule, Invocation.run, initial_eq, runTiles]
    exact congrArg
      (fun next => (exec (engine (graphAlgebra M c plan)) (kernel c hasBias useSoftplus)
        (tileState l (bindBounds s (.input lower) (.input upper)))).bind next)
      (funext ih)

theorem input_dt {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : α)
    (values : Inputs α c) (s : State (Term α))
    (hin : (programIO M c plan hasBias useSoftplus lower upper).InputBinding values s)
    (b t h : Nat) (hb : b < c.batches) (ht : t < c.sequence) (hh : h < c.heads) :
    (s.mem "dt" (b*c.dtBatch+t*c.dtTime+h*c.dtHead)).read .fp32 =
      .input ((logicalInputs c values).dt b t h) := by
  simpa [programIO, memory, inputs, inputAddress, inputMask, logicalInputs, hb, ht, hh,
    mapValue] using hin ⟨0, by simp [programIO, memory, inputs]⟩
      (⟨b,hb⟩,⟨t,ht⟩,⟨h,hh⟩,PUnit.unit) trivial

theorem input_rate {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : α)
    (values : Inputs α c) (s : State (Term α))
    (hin : (programIO M c plan hasBias useSoftplus lower upper).InputBinding values s)
    (h : Nat) (hh : h < c.heads) :
    (s.mem "rate" (h*c.rateHead)).read .fp32 = .input ((logicalInputs c values).rate h) := by
  simpa [programIO, memory, inputs, inputAddress, inputMask, logicalInputs, hh, mapValue] using
    hin ⟨1, by simp [programIO, memory, inputs]⟩ (⟨h,hh⟩,PUnit.unit) trivial

theorem input_bias {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : α)
    (values : Inputs α c) (s : State (Term α))
    (hin : (programIO M c plan hasBias useSoftplus lower upper).InputBinding values s)
    (enabled : hasBias = Bool.true) (h : Nat) (hh : h < c.heads) :
    (s.mem "bias" (h*c.biasHead)).read .real = .input ((logicalInputs c values).bias h) := by
  simpa [programIO, memory, inputs, inputAddress, inputMask, logicalInputs, hh, mapValue] using
    hin ⟨2, by simp [programIO, memory, inputs]⟩ (⟨h,hh⟩,PUnit.unit) enabled

theorem preparation {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : α)
    (values : Inputs α c) (s : State (Term α))
    (hin : (programIO M c plan hasBias useSoftplus lower upper).InputBinding values s)
    (b h k j : Nat) (hb : b < c.batches) :
    interpret M (logicalPrepared c (primitives M c plan) hasBias useSoftplus (.input lower)
      (.input upper) s b h k j) =
      preparedValue M c (logicalInputs c values) hasBias useSoftplus lower upper b h k j := by
  unfold logicalPrepared preparedValue
  split_ifs with active
  · rw [input_dt M c plan hasBias useSoftplus lower upper values s hin b (k*c.chunk+j) h hb active.2.2 active.1]
    cases hasBias
    · exact source_preparation M c plan Bool.false useSoftplus lower upper Bool.true _ _
    · rw [input_bias M c plan Bool.true useSoftplus lower upper values s hin rfl h active.1]
      exact source_preparation M c plan Bool.true useSoftplus lower upper Bool.true _ _
  · norm_num [primitives, graphAlgebra, interpret]

theorem cumulative {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : α)
    (values : Inputs α c) (s : State (Term α))
    (hin : (programIO M c plan hasBias useSoftplus lower upper).InputBinding values s)
    (b h k : Nat) (j : Fin c.BC) (hb : b < c.batches) (hh : h < c.heads) :
    interpret M (logicalCumulative c (primitives M c plan) hasBias useSoftplus
      (.input lower) (.input upper) s b h k j) =
      cumulativeValue M c plan (logicalInputs c values) hasBias useSoftplus lower upper b h k j := by
  simp only [logicalCumulative, primitives, graphAlgebra, source_scan, logicalRate,
    input_rate M c plan hasBias useSoftplus lower upper values s hin h hh,
    interpret, cumulativeValue]
  congr 1
  funext t
  exact preparation M c plan hasBias useSoftplus lower upper values s hin b h k t.val hb

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface

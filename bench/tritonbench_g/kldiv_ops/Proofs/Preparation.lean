import bench.tritonbench_g.kldiv_ops.Proofs.Library

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
noncomputable section

theorem Materialized.trans {Word : Type} {codec : Codec Word} {neg : Word→Word}
    {a b c : Tensor} {s t u : State}
    (ab : Materialized codec neg a b s t) (bc : Materialized codec neg b c t u) :
    Materialized codec neg a c s u := by
  refine ⟨bc.shape.trans ab.shape,bc.device.trans ab.device,ab.preserved.trans bc.preserved,bc.live,?_⟩
  intro i hi
  have hb : i<b.numel := by simpa only [Tensor.numel,ab.shape] using hi
  exact (bc.values i hb).trans (ab.values i hi)

theorem normalize_correct {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (api : Services Word) (lib : Library codec num api) (a b : Tensor) (s t : State)
    (live : a.backing∈s.live) (call : normalize api a s=.ok b t) :
    Materialized codec num.neg a b s t ∧ b.negative=Bool.false := by
  cases hc : api.resolveConj a s with
  | error e failed => simp [normalize,Bind.bind,bind,hc] at call
  | ok resolved ready =>
    have first := lib.conj a s resolved ready live hc
    have hn : api.resolveNeg resolved ready=.ok b t := by
      simpa [normalize,Bind.bind,bind,hc] using call
    have second := lib.neg resolved ready b t first.live hn
    exact ⟨first.trans second.1,second.2⟩

theorem prepare_correct {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (api : Services Word) (lib : Library codec num api) (a b : Tensor) (s t : State)
    (live : a.backing∈s.live) (call : prepare api a s=.ok b t) :
    Materialized codec num.neg a b s t ∧ Dense b := by
  cases hn : normalize api a s with
  | error e failed => simp [prepare,Bind.bind,bind,hn] at call
  | ok normalized ready =>
    have first := normalize_correct codec num api lib a normalized s ready live hn
    have hp : api.contiguous normalized ready=.ok b t := by
      simpa [prepare,Bind.bind,bind,hn] using call
    have second := lib.contiguous normalized ready b t first.1.live first.2 hp
    exact ⟨first.1.trans second.1,second.2⟩

theorem snapshot_correct {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (api : Services Word) (lib : Library codec num api) (a b : Tensor) (s t : State)
    (live : a.backing∈s.live) (call : snapshot api a s=.ok b t) :
    Materialized codec num.neg a b s t ∧ b.negative=Bool.false ∧ (∀ caller,caller∈s.live → b.backing.disjoint caller) := by
  cases hn : normalize api a s with
  | error e failed => simp [snapshot,Bind.bind,bind,hn] at call
  | ok normalized ready =>
    have first := normalize_correct codec num api lib a normalized s ready live hn
    have hp : api.clone normalized ready=.ok b t := by
      simpa [snapshot,Bind.bind,bind,hn] using call
    have second := lib.clone normalized ready b t first.1.live first.2 hp
    exact ⟨first.1.trans second.1,second.2.1,
      fun caller hc=>second.2.2 caller (first.1.preserved.live caller hc)⟩

theorem preserved_value {Word : Type} (codec : Codec Word) (neg : Word→Word)
    {s t : State} (h : Preserves s t) (a : Tensor) (live : a.backing∈s.live) (i : Nat) (hi : i<a.numel) :
    value codec neg t.heap a i=value codec neg s.heap a i :=
  Storage.preserves_value codec neg s.heap t.heap a (h.frame a.backing live) i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host

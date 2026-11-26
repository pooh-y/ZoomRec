"""Independently audit the requested method outputs and metric tables."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from generation_protocol import generation_settings, check_completed_call
from sasrec_candidates import POLICY as CANDIDATE_POLICY
from run_paths import CODE_DIR, HERE, DATA_ROOT, SOURCE


VARIANTS = tuple(json.loads((CODE_DIR / 'ablation_request.json').read_text(encoding='utf-8'))['requested_variants'])
assert VARIANTS and len(set(VARIANTS)) == len(VARIANTS)
assert set(VARIANTS) <= {'full', 'no_adaptive_granularity', 'no_revision', 'no_recovery', 'no_both'}
KS = (1, 3, 5, 10)
METRICS = tuple([f"HR@{k}" for k in KS] + [f"NDCG@{k}" for k in KS])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metric_table(ranks):
    n = len(ranks)
    if not n:
        raise ValueError("cannot compute metrics over zero ranks")
    out = {f"HR@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out.update({f"NDCG@{k}": math.fsum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / n
                for k in KS})
    return out


def check_calls(result_dir, actions):
    totals = collections.Counter()
    for name, flag, limit in (("plan_calls", "plan_called", 256),
                              ("decision_calls", "decision_called", 2048)):
        call_ids = []
        for action in actions:
            ids = action.get(name[:-1] + "_ids", [action["event_id"]] if action[flag] else [])
            assert bool(ids) == action[flag]
            call_ids.extend(ids)
        expected = set(call_ids)
        assert len(call_ids) == len(expected)
        path = result_dir / f"{name}.jsonl"
        # run_variant does not create a call log when this stage has no requests.
        # A required missing log or an unexpected cached call must still fail.
        calls = rows(path) if path.exists() or expected else []
        by_id = {call["id"]: call for call in calls}
        assert len(by_id) == len(calls) and set(by_id) == expected
        for call in calls:
            generation = call["generation"]
            assert generation == generation_settings(limit)
            check_completed_call(call,generation)
            assert 0 < call["output_tokens"] <= call['attempt_token_limit']
            assert call["input_tokens"] <= 12000
        totals[name] = len(calls)
    return totals


def check_trace(action, event):
    index = event["index"]
    assert len(event["history"]) == index
    prior = action["pre_observation"]
    post = action["post_observation"]
    assert prior["events"] == index and post["events"] == index + 1
    assert post["as_of"] >= prior["as_of"]
    assert action["context"]["history_length"] == index
    recent = action["context"]["recent_history"]
    assert len(recent) == min(index, 5)
    start = index - len(recent)
    for row, hist_index in zip(recent, range(start, index)):
        assert row["index"] == hist_index and row["item"] == event["history"][hist_index]
        assert row["timestamp"] <= prior["as_of"]
    for hypothesis in action["observations"]["retrieved"]:
        for key in ("source_evidence", "target_evidence"):
            for evidence in hypothesis.get(key, []):
                assert evidence["index"] < index
                assert event["history"][evidence["index"]] == evidence["item"]
                assert evidence["timestamp"] <= prior["as_of"]


def check_mechanisms(action, variant):
    adaptive = variant not in ("no_adaptive_granularity", "no_both")
    revision = variant not in ("no_revision", "no_both")
    recovery = variant not in ("no_recovery", "no_both")
    for trace_name in ("pre_observation", "post_observation"):
        trace = action[trace_name]
        assert trace["variant"] == variant
        assert trace["mechanisms"] == dict(adaptive_granularity=adaptive,
            local_revision=revision, recovery=recovery)
        if not revision:
            assert not trace["revisions"]
            assert all(h["content"] == h["original_content"] for h in trace["hypotheses"])
        if not recovery:
            withdrawn = {tuple(key) for key in trace["permanently_withdrawn"]}
            for transition in trace["transitions"]:
                assert transition["operation"] != "recover"
                if transition["after"] == "withdrawn":
                    assert tuple(transition["key"]) in withdrawn
            for hypothesis in trace["hypotheses"]:
                if tuple(hypothesis["key"]) in withdrawn:
                    assert hypothesis["assessment"]["state"] == "withdrawn"
                    assert hypothesis["assessment"]["effective_weight"] == 0


def reconstruct_order(case, action, prediction, execution, setting):
    candidates = case["candidates"]
    baseline = case["baseline"]
    assert len(candidates) == len(set(candidates)) == len(baseline) == 20
    assert all(math.isfinite(float(score)) for score in baseline)
    base = dict(zip(candidates, map(float, baseline)))
    original = sorted(base, key=lambda item: (-base[item], item))
    order = list(original)
    bonuses = {int(item): float(value) for item, value in execution["bonuses"].items()}
    assert set(bonuses) <= set(base)
    assert all(math.isfinite(value) for value in bonuses.values())
    approved = action["approved"]
    assert prediction["n_approved"] == len({row["item"] for row in approved})
    assert prediction.get("n_approved_claims", len(approved)) == len(approved)
    assert len({(row["item"], row["hypothesis"]) for row in approved}) == len(approved)
    expected_bonuses = {}
    claim_bonus = {}
    for row in approved:
        assert row["item"] in base
        if setting["min_confidence"] == "high" and row["confidence"] != "high":
            continue
        value = setting["strength"] * row["ledger_weight"]
        if value <= 0 or row["confidence"] not in ("high", "medium"):
            continue
        claim_bonus[row["item"], row["hypothesis"]] = value
        expected_bonuses[row["item"]] = max(expected_bonuses.get(row["item"], value), value)
    assert set(bonuses) == set(expected_bonuses)
    for item, value in expected_bonuses.items():
        assert math.isclose(float(bonuses[item]), value, rel_tol=0, abs_tol=1e-12)
    for crossing in execution["crossings"]:
        promoted, displaced = crossing["promoted"], crossing["displaced"]
        assert promoted in base and displaced in base and promoted != displaced
        source_pos = crossing["from_rank"] - 1
        target_pos = crossing["to_rank"] - 1
        assert target_pos == source_pos - 1 and source_pos > 0
        assert order[source_pos] == promoted and order[target_pos] == displaced
        assert math.isclose(crossing["original_score_gap"], base[displaced] - base[promoted],
                            rel_tol=0, abs_tol=1e-10)
        key = (promoted, crossing["hypothesis"])
        assert key in claim_bonus, "crossing requires an unused qualifying claim"
        bonus = claim_bonus.pop(key)
        assert math.isclose(crossing["bonus"], bonus, rel_tol=0, abs_tol=1e-12)
        assert (-(base[promoted] + bonus), promoted) < (
            -(base[displaced] + bonuses.get(displaced, 0.)), displaced)
        order[target_pos], order[source_pos] = order[source_pos], order[target_pos]
    moves = max(abs(i - original.index(item)) for i, item in enumerate(order))
    assert moves == execution["max_displacement"]
    assert prediction["ranking_changed"] == bool(execution["crossings"])
    return order


def audit(allow_partial=False):
    manifest = read(HERE / "manifest.json")
    events = rows(SOURCE / "events.jsonl")
    assert manifest['protocol'] == 'leave-one-out-v1'
    assert manifest.get('candidate_policy') == CANDIDATE_POLICY
    assert len(events) == manifest["counts"]["test_events"] == manifest['counts']['users']
    sequences = read(DATA_ROOT / 'freeze/sequences.json')
    assert {e['uid'] for e in events} == set(sequences)
    for event in events:
        items = sequences[event['uid']]['items']
        assert event['index'] == len(items) - 1
        assert event['history'] == items[:-1] and event['target'] == items[-1]
    assert manifest["metrics"] == list(METRICS)
    assert set(manifest["variants"]) == set(VARIANTS)
    setting = manifest["selection"]
    shards = sorted((HERE / "shards").glob("[0-9][0-9][0-9][0-9]"))
    completed = [p for p in shards if (p / "done.json").exists()]
    if not allow_partial:
        assert len(completed) == manifest["n_shards"]
    verified = collections.Counter()
    all_ranks = {variant: [] for variant in VARIANTS}

    for folder in completed:
        shard_id = int(folder.name)
        case_path = SOURCE / "shards" / folder.name / "cases.jsonl"
        cases = rows(case_path)
        start = shard_id * manifest["chunk_size"]
        expected_events = events[start:start + manifest["chunk_size"]]
        assert len(cases) == len(expected_events)
        assert read(folder / "done.json")["n_events"] == len(cases)
        for variant in VARIANTS:
            result_dir = folder / variant
            predictions = rows(result_dir / "predictions.jsonl")
            actions = rows(result_dir / "actions.jsonl")
            executions = rows(result_dir / "ranking_executions.jsonl")
            assert len(predictions) == len(actions) == len(executions) == len(cases)
            calls = check_calls(result_dir, actions)
            verified[f"{variant}:plan_calls"] += calls["plan_calls"]
            verified[f"{variant}:decision_calls"] += calls["decision_calls"]
            ranks = []
            for case, event, prediction, action, execution in zip(
                    cases, expected_events, predictions, actions, executions):
                identity = ("event_id", "uid", "index", "history", "target", "candidates", "sasrec_scores")
                assert all(case[key] == event[key] == prediction[key] for key in identity)
                assert action["event_id"] == execution["event_id"] == event["event_id"]
                assert action["method"] == execution["method"] == prediction["method"] == variant
                check_trace(action, event)
                check_mechanisms(action, variant)
                order = reconstruct_order(case, action, prediction, execution, setting)
                rank = next((pos for pos, item in enumerate(order, 1) if item == event["target"]), None)
                assert prediction["rank"] == rank
                assert prediction["top10"] == order[:10]
                assert len(set(prediction["top10"])) == 10
                ranks.append(rank)
                verified[variant] += 1
            stored = read(result_dir / "metrics.json")
            computed = metric_table(ranks)
            assert set(stored) == set(METRICS)
            for name in METRICS:
                assert math.isclose(stored[name], computed[name], rel_tol=0, abs_tol=1e-12)
            done_metrics = read(folder / "done.json")["metrics"][variant]
            for name in METRICS:
                assert math.isclose(done_metrics[name], computed[name], rel_tol=0, abs_tol=1e-12)
            all_ranks[variant].extend(ranks)

    report = dict(status="passed", completed_shards=len(completed), total_shards=manifest["n_shards"],
        complete=not allow_partial, checked_events_by_variant=dict(verified),
        checks=["frozen event and candidate identity", "prior-only observation timing",
                "baseline plus adjacent-swap rank reconstruction", "one adjacent swap per triggering claim",
                "uniform inference configuration and exact call coverage",
                "disabled revision preserves content and disabled recovery preserves withdrawal",
                "per-shard metrics independently recomputed"])
    if not allow_partial:
        result = read(HERE / "results.json")
        assert result["status"] == "complete"
        assert result["n_events"] == len(events)
        assert result["manifest"] == manifest
        assert set(result["metrics"]) == set(VARIANTS)
        for path, expected_hash in manifest["source_sha256"].items():
            assert digest(CODE_DIR / path) == expected_hash, path
        source_files = manifest['source_preservation']['sha256']
        for name, expected in source_files.items():
            assert digest(CODE_DIR / name) == expected, name
        report["frozen_hashes_checked"] = len(manifest["source_sha256"])
        report["package_source_files_unchanged"] = len(source_files)
        report["auditor_sha256"] = digest(Path(__file__))
        final_tables = {}
        for variant in VARIANTS:
            assert len(all_ranks[variant]) == len(events)
            calculated = metric_table(all_ranks[variant])
            final_tables[variant] = calculated
            for name in METRICS:
                assert math.isclose(result["metrics"][variant][name], calculated[name],
                                    rel_tol=0, abs_tol=1e-12)
        report["final_metrics_recomputed"] = True
        report["metrics"] = final_tables
        (HERE / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    audit(parser.parse_args().allow_partial)

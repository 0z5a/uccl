import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'
manifest = json.loads((ROOT / 'source-manifest.json').read_text())
rows, samples = [], []


def metrics(times: list[float]) -> dict[str, float]:
    return dict(median_ms=statistics.median(times),
        p95_ms=statistics.quantiles(times, n=100, method='inclusive')[94],
        min_ms=min(times), max_ms=max(times))


for device in (0, 1):
    variants, waves = {}, {wave: {} for wave in range(2)}
    for variant in ('base', 'v1', 'v2'):
        times = []
        for wave in range(2):
            records = [json.loads(line) for line in
                (EVIDENCE / f'bench-{wave}-{variant}-{device}.log').read_text().splitlines()
                if line.startswith('{')]
            assert len(records) == 7 and [r['round'] for r in records] == list(range(7))
            for record in records:
                assert record['pass'] and record['device'] == device
                assert record['via'] == 'adapter' and record['group'] == 'warp'
                assert record['groups'] == 64 and record['queues'] == 32
                assert record['batch_size'] == 2048 and record['iterations'] == 2
                assert record['bytes'] == 262144 and record['hidden_size'] == 2048
                assert record['capacity'] == 4096 and record['source_reuse_checked']
                assert record['shared'] and record['channel_hint'] == 'lane'
                assert record['num_lanes'] == 1 and record['delay_us'] == 0
                assert record['delay_queue'] == 31 and not record['consumer_diagnostics']
                assert all(record[field] == 0 for field in ('callback_ns',
                    'copy_event_submit_ns', 'event_sync_ns', 'pattern_check_ns',
                    'between_callbacks_ns'))
                assert record['writes'] == 4096
                assert record['quiet'] == 4096 * (32 if variant == 'base' else 1)
                if record['round'] >= 2:
                    times.append(record['elapsed_ms'])
                    samples.append(dict(variant=variant, wave=wave,
                        physical_gpu=0 if device == 0 else 3, **record))
            waves[wave][variant] = metrics([r['elapsed_ms'] for r in records[2:]])
        assert len(times) == 10
        variants[variant] = metrics(times)
    for arms in waves.values():
        arms['original_over_v2'] = arms['base']['median_ms'] / arms['v2']['median_ms']
        arms['v1_over_v2'] = arms['v1']['median_ms'] / arms['v2']['median_ms']
    rows.append(dict(physical_gpu=0 if device == 0 else 3,
        fixture_sha256=manifest['fixture_sha256'], independent_starts_per_arm=2,
        warm_observations_per_start=5, warm_observations_per_arm=10,
        variants=variants, waves=waves,
        original_over_v2=variants['base']['median_ms'] / variants['v2']['median_ms'],
        v1_over_v2=variants['v1']['median_ms'] / variants['v2']['median_ms']))
(ROOT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
(ROOT / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in samples))
print('| GPU | Wave/order | Starts / warm rounds per arm | Original median/p95 | V1 median/p95 | V2 median/p95 | Original/V2 | V1/V2 |')
print('| --- | --- | --- | --- | --- | --- | --- | --- |')
for row in rows:
    for label, arms, starts, count in [
        ('pooled ABC/CBA', row['variants'], 2, 10),
        ('0: ABC', row['waves'][0], 1, 5),
        ('1: CBA', row['waves'][1], 1, 5),
    ]:
        cells = [f"{arms[variant]['median_ms']:.3f} / {arms[variant]['p95_ms']:.3f}"
                 for variant in ('base', 'v1', 'v2')]
        print(f"| {row['physical_gpu']} | {label} | {starts} / {count} | " +
            ' | '.join(cells) + f" | {arms['base']['median_ms'] / arms['v2']['median_ms']:.2f}×" +
            f" | {arms['v1']['median_ms'] / arms['v2']['median_ms']:.2f}× |")

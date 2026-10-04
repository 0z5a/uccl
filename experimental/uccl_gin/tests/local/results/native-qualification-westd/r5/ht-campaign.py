"""One explicitly admitted native HT/GIN correctness stage; no installations."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--grant', type=Path, required=True)
parser.add_argument('--stage', choices=('build', 'test', 'all'), required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
grant = json.loads(args.grant.read_text())
assert grant['thread'] == '01a104db-113f-7890-85a0-ddc427410d08'
assert grant['status'] == 'granted' and grant['scope'] == 'native_ht_and_gin_correctness'
assert grant['machine'] in ('thor', 'rtx5080', 'westd-5090')
assert grant['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert str(grant['init_start_ticks']) == Path('/proc/1/stat').read_text().rsplit(') ', 1)[1].split()[19]
assert root.resolve() == Path(grant['work']).resolve()
assert grant['exclusive_gpu_and_io'] and grant['runtime_existing_readonly']
inherited = os.environ.pop('UCCL_NATIVE_LOCK_FDS', '')
locks = ([os.fdopen(int(fd), 'a') for fd in inherited.split(':')] if inherited else
         [Path(path).open('a') for path in grant['locks']])
assert len(locks) >= (1 if grant['machine'] == 'thor' else 2)
assert len(locks) == len(grant['locks'])
for lock, path in zip(locks, grant['locks']):
    assert (os.fstat(lock.fileno()).st_dev, os.fstat(lock.fileno()).st_ino) == (
        Path(path).stat().st_dev, Path(path).stat().st_ino)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
fds = tuple(lock.fileno() for lock in locks)
filesystem = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(root)], text=True).strip()
assert 'nfs' not in filesystem.lower()
observed = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used',
                                   '--format=csv,noheader'], text=True)
device_row = next(line for line in observed.splitlines() if line.split(',')[0].strip() == str(grant['gpu_index']))
assert grant['gpu_uuid'] in device_row
evidence = root / 'evidence' / ('ht-build' if args.stage == 'all' else 'ht-' + args.stage)
assert not evidence.exists(), 'inspect actual terminal state before preparing a distinct repair run'
evidence.mkdir(parents=True)
manifest = json.loads((root / 'source-ht-manifest.json').read_text())
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = str(grant['gpu_index'])
env['PYTHONUNBUFFERED'] = '1'
env['HF_HUB_OFFLINE'] = '1'
local = root / 'source/native-ht/experimental/uccl_gin/tests/local'
sm = str(grant['sm'])
assert sm in ('110', '110a', '120')
binary = local / ('build/sm' + sm)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name: str, command: list[str]) -> None:
    record = {'case': name, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}
    with (evidence / (name + '.log')).open('w') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, pass_fds=fds)
        record['pid'] = process.pid
        (evidence / 'live.json').write_text(json.dumps(record, indent=2) + '\n')
        record['exit'] = process.wait()
    record['ended_utc'] = datetime.now(timezone.utc).isoformat()
    with (evidence / 'commands.jsonl').open('a') as output:
        output.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
    assert record['exit'] == 0, 'preserve the failure; do not hide it with retries'


(evidence / 'admission.json').write_text(json.dumps({'grant': grant, 'filesystem': filesystem,
    'observed_gpus': observed, 'controller_pid': os.getpid(),
    'controller_start_ticks': Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
    'controller_sha256': sha(Path(__file__)), 'manifest_sha256': sha(root / 'source-ht-manifest.json')}, indent=2) + '\n')
if args.stage in ('build', 'all'):
    assert sha(root / 'source-ht.tar.gz') == manifest['archive_sha256']
    assert sha(root / 'nccl-include.tar.gz') == manifest['nccl_archive_sha256']
    with tarfile.open(root / 'source-ht.tar.gz') as archive:
        archive.extractall(root / 'source', filter='data')
    with tarfile.open(root / 'nccl-include.tar.gz') as archive:
        archive.extractall(root, filter='data')
    for path, expected in manifest['files'].items():
        assert sha(root / 'source/native-ht' / path) == expected
    for path, expected in manifest['nccl_files'].items():
        assert sha(root / 'nccl-include' / path) == expected
    numa = json.loads((root / 'numactl-source-manifest.json').read_text())
    for path, expected in numa['files'].items():
        assert sha(root / 'numactl-v2.0.14' / path) == expected['sha256']
    run('toolchain', [grant['cuda_home'] + '/bin/nvcc', '--version'])
    command = ['make', '-C', str(local), '-j1', f'SM={sm}', f'CUDA_HOME={grant["cuda_home"]}',
               f'NCCL_INCLUDE_DIR={root / "nccl-include"}', *grant['build_flags'],
               'all', 'adapter-tests', 'adapter-compile-fail', 'model-tests', 'ht-tests']
    run('native-build', command)
    (evidence / 'binary-hashes.json').write_text(json.dumps({name: sha(binary / name) for name in
        ('capabilities', 'queue_smoke', 'adapter_signal', 'libmodel_transport.so',
         'ht_single_rank-nccl', 'ht_single_rank-uccl')}, indent=2) + '\n')
if args.stage in ('test', 'all'):
    if args.stage == 'all':
        admission = (evidence / 'admission.json').read_text()
        (evidence / 'workloads-finished.json').write_text(json.dumps({'status': 'build_ready',
            'same_controller_continues_with_locks_held': True}) + '\n')
        evidence = root / 'evidence/ht-test'
        evidence.mkdir()
        (evidence / 'admission.json').write_text(admission)
    hashes = json.loads((root / 'evidence/ht-build/binary-hashes.json').read_text())
    for name, expected in hashes.items():
        assert sha(binary / name) == expected
    run('capabilities', [str(binary / 'capabilities'), '--device', '0'])
    run('queue-smoke', [str(binary / 'queue_smoke'), '--device', '0', '--producers', '32',
                        '--capacity', '512', '--commands', '100000', '--rounds', '3'])
    run('adapter-signal', [str(binary / 'adapter_signal'), '--device', '0'])
    run('native-tensor', [grant['python'], str(local / 'model_e2e.py'), '--library',
                          str(binary / 'libmodel_transport.so'), '--arm', 'v2', '--native-only'])
    for batch, concurrency, tail in ((8, 32, 0), (16, 64, 0), (32, 128, 0), (64, 128, 0), (8, 32, 4)):
        for backend in ('nccl', 'uccl'):
            run(f'ht-{backend}-B{batch}C{concurrency}-tail{tail}', [
                str(binary / ('ht_single_rank-' + backend)), '--batch', str(batch),
                '--concurrency', str(concurrency), '--token-tail', str(tail), '--rounds', '7'])
    rows = [json.loads(line) for path in evidence.glob('ht-*.log')
            for line in path.read_text().splitlines() if line.startswith('{')]
    assert len(rows) == 70
    for batch, concurrency, tail in ((8, 32, 0), (16, 64, 0), (32, 128, 0), (64, 128, 0), (8, 32, 4)):
        arms = [[row for row in rows if row['backend'] == backend and row['batch'] == batch
                 and row['concurrency'] == concurrency and row['tokens_per_batch'] == batch * 128 - tail]
                for backend in ('nccl', 'uccl')]
        for arm in arms:
            assert [row['round'] for row in arm] == list(range(7))
            assert [row['warmup'] for row in arm] == [True, True, False, False, False, False, False]
            assert all(row['pass'] and row['gin_commands'] == 0 and row['completed_requests'] == concurrency
                       and row['peak_pending_requests'] == concurrency and row['actual_batches'] == concurrency // batch
                       for row in arm)
        assert [row['oracle_hash'] for row in arms[0]] == [row['oracle_hash'] for row in arms[1]]
    (evidence / 'native-summary.json').write_text(json.dumps({'pass': True, 'machine': grant['machine'],
        'native_ht_processes': 10, 'native_ht_rounds': 70, 'cases': 5, 'backends': 2,
        'scope': 'actual single-rank HT CUDA kernels and GIN local tensor/signal gates; no model/NIC/timing claim'}, indent=2) + '\n')
(evidence / 'workloads-finished.json').write_text(json.dumps({'status': 'results_ready',
    'stage': args.stage, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'requires_actual_controller_exit_and_evidence_collection': True}, indent=2) + '\n')

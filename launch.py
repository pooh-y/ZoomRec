"""Prepare ZoomRec, own a local inference server, run and audit resumable workers."""
import argparse, json, os, signal, subprocess, sys, tempfile, time, urllib.request
from contextlib import nullcontext
from pathlib import Path

sys.dont_write_bytecode = True
from generation_protocol import MAX_MODEL_LEN, GUIDED_BACKEND
from run_paths import CODE_DIR, HERE, MODEL_PATH, QWEN_PORT, CHECKPOINT_MODE, project_reference

RUNTIME=Path(sys.executable)
PORT=QWEN_PORT

def child_log(run_dir, name, transient):
    # Transient runs stream diagnostics directly to the caller.
    return nullcontext(None) if transient else (run_dir / name).open('a')


def stop_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.48)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--pilot',action='store_true',help='Complete one formal shard before full launch.')
    parser.add_argument('--no-save', action='store_true',
        help='Keep run data in RAM and discard it on exit. This is the default unless ZOOMREC_RUN_DIR is set.')
    parser.add_argument('--checkpoint',choices=['bundled','fresh'],default=CHECKPOINT_MODE)
    args=parser.parse_args()
    if args.checkpoint == 'bundled':
        print('The published checkpoint was selected using a different validation protocol. '
              'Using Top-20 training in the run directory.', flush=True)
        args.checkpoint = 'fresh'
    if args.workers < 1 or not 0 < args.gpu_memory_utilization < 1:
        parser.error('workers must be positive and GPU memory utilization must be between 0 and 1')
    transient = args.no_save or not os.environ.get('ZOOMREC_RUN_DIR')
    if args.detach:
        cmd=[str(RUNTIME),'-u',str(CODE_DIR/'launch.py'),'--gpu',str(args.gpu),'--workers',str(args.workers),
             '--gpu-memory-utilization',str(args.gpu_memory_utilization),'--checkpoint',args.checkpoint]
        if args.pilot: cmd.append('--pilot')
        if transient:
            cmd.append('--no-save')
            log_path = Path(os.devnull)
        else:
            HERE.mkdir(parents=True, exist_ok=True)
            log_path = HERE / 'supervisor.log'
        with log_path.open('a') as log:
            p=subprocess.Popen(cmd,cwd=CODE_DIR if transient else HERE,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=p.pid))); return
    if not transient:
        execute(args, HERE, transient=False)
        return
    previous = signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        with tempfile.TemporaryDirectory(prefix='zoomrec-run-', dir='/dev/shm') as temporary:
            print(f'Temporary run directory {project_reference(temporary)}', flush=True)
            execute(args, Path(temporary), transient=True)
    finally:
        signal.signal(signal.SIGTERM, previous)


def execute(args, run_dir, transient):
    import fcntl

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='20261101',OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4',ZOOMREC_CHECKPOINT=args.checkpoint,ZOOMREC_RUN_DIR=str(run_dir),
            ZOOMREC_QWEN_URL=f'http://127.0.0.1:{PORT}/v1/completions')
        if transient:
            tmp = run_dir / 'tmp'
            tmp.mkdir()
            cache = run_dir / 'cache'
            env.update(TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(cache),
                       XDG_CONFIG_HOME=str(run_dir / 'config'), VLLM_NO_USAGE_STATS='1',
                       VLLM_CONFIG_ROOT=str(run_dir / 'config' / 'vllm'),
                       VLLM_CACHE_ROOT=str(cache / 'vllm'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                       TRITON_CACHE_DIR=str(cache / 'triton'), CUDA_CACHE_PATH=str(cache / 'cuda'),
                       TORCH_EXTENSIONS_DIR=str(cache / 'extensions'))
        # Fit/verify SASRec before occupying GPU memory for the inference server.
        with child_log(run_dir, 'prepare.log', transient) as prepare_log:
            subprocess.run([str(RUNTIME),'-u',str(CODE_DIR/'prepare_loo.py'),'--device','cuda:0'],
                cwd=CODE_DIR,env=dict(env,CUDA_VISIBLE_DEVICES=str(args.gpu)),check=True,
                stdout=prepare_log,stderr=subprocess.STDOUT)
        # Refuse to attach to an unowned server at the same port.
        import socket
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',PORT))==0:
                raise RuntimeError('Dedicated inference port is already in use')
        cmd=[str(RUNTIME),'-m','vllm.entrypoints.openai.api_server',
            '--model',str(MODEL_PATH),
            '--served-model-name','beauty-qwen','--host','127.0.0.1','--port',str(PORT),
            '--dtype','bfloat16','--seed','20261101','--max-model-len',str(MAX_MODEL_LEN),
            '--gpu-memory-utilization',str(args.gpu_memory_utilization),'--max-num-seqs','64','--max-num-batched-tokens','8192',
            '--guided-decoding-backend',GUIDED_BACKEND,
            '--disable-log-requests','--generation-config','vllm']
        with child_log(run_dir, 'inference_server.log', transient) as log:
            server=subprocess.Popen(cmd,cwd=run_dir,env=dict(env,CUDA_VISIBLE_DEVICES=str(args.gpu)),
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        # Record the reproducible command using paths relative to the project.
        server_command = list(cmd)
        server_command[0] = RUNTIME.name
        server_command[server_command.index('--model') + 1] = project_reference(MODEL_PATH)
        processes=[]
        try:
            (run_dir/'pids.json').write_text(json.dumps(dict(supervisor=os.getpid(),server=server.pid,
                server_command=server_command,server_command_base='project',workers=[],gpu=args.gpu,port=PORT)),encoding='utf-8')
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Inference server exited with code {server.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=3) as r:
                        if r.status==200: break
                except Exception: time.sleep(3)
            else: raise TimeoutError('Inference server startup timed out')
            # Summaries use only observed histories. Scoring happens after they are ready.
            with child_log(run_dir, 'prepare.log', transient) as prepare_log:
                for arguments in [('history_states.py',),
                                  ('full_evaluation.py','prepare','--device','cuda:0')]:
                    subprocess.run([str(RUNTIME),'-u',str(CODE_DIR/arguments[0]),*arguments[1:]],
                        cwd=CODE_DIR,env=dict(env,CUDA_VISIBLE_DEVICES=str(args.gpu)),check=True,
                        stdout=prepare_log,stderr=subprocess.STDOUT)
            workers=1 if args.pilot else args.workers
            for i in range(workers):
                cmd=[str(RUNTIME),'-u',str(CODE_DIR/'full_evaluation.py'),'worker','--worker',str(i),'--workers',str(workers)]
                if args.pilot: cmd+=['--limit','1']
                with child_log(run_dir, f'worker_{i}.log', transient) as worker_log:
                    p=subprocess.Popen(cmd,cwd=run_dir,env=env,stdout=worker_log,stderr=subprocess.STDOUT)
                processes.append(p)
            (run_dir/'pids.json').write_text(json.dumps(dict(supervisor=os.getpid(),server=server.pid,
                server_command=server_command,server_command_base='project',workers=[p.pid for p in processes],gpu=args.gpu,port=PORT)),encoding='utf-8')
            codes=[p.wait() for p in processes]
            if any(codes): raise RuntimeError('Worker failures: '+str(codes))
            if not args.pilot:
                subprocess.run([str(RUNTIME),'-u',str(CODE_DIR/'full_evaluation.py'),'aggregate'],cwd=run_dir,env=env,check=True)
            audit_cmd=[str(RUNTIME),'-u',str(CODE_DIR/'audit_ablation_outputs.py')]
            if args.pilot: audit_cmd.append('--allow-partial')
            subprocess.run(audit_cmd,cwd=run_dir,env=env,check=True)
        finally:
            for p in processes:
                if p.poll() is None: p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGKILL)
                    server.wait()

if __name__=='__main__': main()

import os
import sys
import argparse
import subprocess
import importlib.util
import json

def run_qwen(repo_dir):
    script = os.path.join(repo_dir, 'experiments', 'run_lightmem_qwen.py')
    env = os.environ.copy()
    src_path = os.path.join(repo_dir, 'src')
    prev = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = src_path if not prev else f"{src_path}{os.pathsep}{prev}"
    env_path = os.path.join(repo_dir, '.env')
    if os.path.exists(env_path):
        for line in open(env_path, 'r', encoding='utf-8').read().splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k, v = s.split('=', 1)
            v = v.strip().strip('"').strip('\'')
            env.setdefault(k.strip(), v)
    subprocess.run([sys.executable, script], check=True, env=env)

def run_gpt(repo_dir):
    script = os.path.join(repo_dir, 'experiments', 'run_lightmem_gpt.py')
    env = os.environ.copy()
    src_path = os.path.join(repo_dir, 'src')
    prev = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = src_path if not prev else f"{src_path}{os.pathsep}{prev}"
    env_path = os.path.join(repo_dir, '.env')
    if os.path.exists(env_path):
        for line in open(env_path, 'r', encoding='utf-8').read().splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k, v = s.split('=', 1)
            v = v.strip().strip('"').strip('\'')
            env.setdefault(k.strip(), v)
    subprocess.run([sys.executable, script], check=True, env=env)

def write_summary(repo_dir):
    results_dir = os.path.join(repo_dir, 'results')
    qdrant_dir = os.path.join(repo_dir, 'qdrant_data')
    out_path = os.path.join(repo_dir, 'reports', 'summary.json')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(qdrant_dir, exist_ok=True)
    spec = importlib.util.spec_from_file_location('summarize_results', os.path.join(repo_dir, 'scripts', 'summarize_results.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = mod.summarize(results_dir, qdrant_dir)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"total_samples={summary['total_samples']}")
    print(f"correct_count={summary['correct_count']}")
    print(f"accuracy={summary['accuracy']:.4f}")
    print(f"avg_construction_time={summary['avg_construction_time']:.3f}s")
    print(f"total_vectors={summary['total_vectors']}")
    print(f"avg_vectors_per_collection={summary['avg_vectors_per_collection']:.2f}")
    print(f"output={out_path}")

def choose_target():
    print('请选择要运行的测设:')
    print('1) qwen')
    print('2) gpt')
    print('3) both')
    choice = input('输入编号并回车: ').strip()
    if choice == '1':
        return 'qwen'
    if choice == '2':
        return 'gpt'
    if choice == '3':
        return 'both'
    return 'qwen'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['qwen', 'gpt', 'both', 'summarize_only'])
    args = parser.parse_args()
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    target = args.target or choose_target()
    if target == 'qwen':
        run_qwen(repo_dir)
        write_summary(repo_dir)
        return
    if target == 'gpt':
        run_gpt(repo_dir)
        write_summary(repo_dir)
        return
    if target == 'both':
        run_qwen(repo_dir)
        run_gpt(repo_dir)
        write_summary(repo_dir)
        return
    if target == 'summarize_only':
        write_summary(repo_dir)
        return

if __name__ == '__main__':
    main()

import argparse
from pathlib import Path
from semantic_memory.integrations.lwow import DEFAULT_REPO, load_original_functions, official_graph_path


def main():
    p = argparse.ArgumentParser(description='Execute the original LWOW graph loader and summary')
    p.add_argument('--repo', type=Path, default=DEFAULT_REPO, help='Path to unmodified LWOW clone')
    p.add_argument('--model', choices=['Mistral', 'Llama3', 'Haiku'], default='Mistral')
    args = p.parse_args()

    source = (args.repo / 'reproducibility' / 'FA_Functions.py').resolve()
    edge_list = official_graph_path(args.repo, args.model).resolve()
    if not edge_list.exists():
        p.error(f'Missing published original edge list: {edge_list}; see README.md')

    funcs = load_original_functions(args.repo)
    graph = funcs.edgelist2graph(str(edge_list), directed=False)
    summary = funcs.graph_summary({args.model: graph})

    print('Source:', source)
    print('Network:', edge_list)
    print(summary.to_string())


if __name__ == '__main__':
    main()

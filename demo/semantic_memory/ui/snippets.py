import inspect
from pathlib import Path

from ..analysis.bias import bias_evaluation
from ..analysis.networks import build_association_layer
from ..integrations.lwow import load_original_functions

SNIPPETS = {
    'FA_edgeList': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Turn records into response pairs.',
    },
    'graphFromEdgeList': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Count repeated responses to build directed, weighted links.',
    },
    'makeUndirected': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Transform into undirected graph, retaining the larger weight when bidir edges merged.',
    },
    'WNfilter': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Filter out words not recognized by WordNet.',
    },
    'idiosynfilter': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Remove weight-one edges - exclusively for free associations.',
    },
    'CC': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Extract the largest connected component.',
    },
    'build_association_layer': {
        'origin': 'orchestration wrapper', 'language': 'python',
        'purpose': 'Interpretation of the flow used when constructing the free associations layer.',
    },
    'edgelist2graph': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Load a previously-constructed network.',
    },
    'spreadSimple': {
        'origin': 'original LWOW R source code', 'language': 'r',
        'purpose': 'Original FA_spreadr.r function.',
    },
    'normalizeDF': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Normalize the complete matrix before selecting any display rows.',
    },
    'matricesGender': {
        'origin': 'original LWOW source code', 'language': 'python',
        'purpose': 'Select matched target rows and female/male prime columns from normalized activation.',
    },
    'bias_evaluation': {
        'origin': 'interpretation', 'language': 'python',
        'purpose': 'Interpretation based on the article\'s descriptions.',
    },
}


def snippet_source(repo: Path, name: str) -> str:
    if name not in SNIPPETS:
        raise ValueError(f'Unknown code snippet: {name}')

    if name == 'spreadSimple':
        text = (Path(repo) / 'reproducibility/FA_spreadr.r').read_text(encoding='utf-8')
        start = text.index('spreadSimple <- function(')
        end = text.index('\n# Load graphs', start)
        return text[start:end].rstrip()

    if name == 'build_association_layer':
        function = build_association_layer
    elif name == 'bias_evaluation':
        function = bias_evaluation
    else:
        function = getattr(load_original_functions(Path(repo)), name)

    return inspect.getsource(function)

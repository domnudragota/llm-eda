import re

import networkx as nx
import nltk
from nltk.corpus import wordnet as wn

from ..config import DATA_DIR

LOCAL_CORPUS = DATA_DIR / 'reference' / 'nltk_data'
nltk.data.path.insert(0, str(LOCAL_CORPUS))

FUNCTION_WORDS = frozenset('''a an the and or but of to in on at by for from with
without as is are be been being was were that which who whose this these those
it its they their one any some all other such especially usually used having
have has can may not also into through than either both relating'''.split())


def senses(cue: str):
    return wn.synsets(cue.replace(' ', '_'))


def build_reference_graph(cue: str, sense_name: str, layer: str) -> nx.Graph:
    if layer not in {'definition', 'category'}:
        raise ValueError('Unknown WordNet reference layer')
    sense = wn.synset(sense_name)
    if sense not in senses(cue):
        raise ValueError('Selected sense does not belong to the cue')
    graph = nx.Graph(source='WordNet ' + wn.get_version(), layer=layer,
                     sense=sense.name(), definition=sense.definition())
    graph.add_node(cue)
    evidence = []
    if layer == 'definition':
        tokens = sorted(set(re.findall(r'[a-z]+', sense.definition().lower())))
        for token in tokens:
            if len(token) >= 3 and token not in FUNCTION_WORDS and wn.synsets(token):
                evidence.append((token, 'definition token', sense.name()))
    else:
        for lemma in sense.lemmas():
            evidence.append((lemma.name().replace('_', ' '), 'synonym', sense.name()))
            if lemma.name().lower() == cue.replace(' ', '_').lower():
                for antonym in lemma.antonyms():
                    evidence.append((antonym.name().replace('_', ' '), 'antonym',
                                     antonym.synset().name()))
        for relation, related in [('hypernym', sense.hypernyms()),
                                  ('hyponym', sense.hyponyms())]:
            for related_sense in related:
                for lemma in related_sense.lemmas():
                    evidence.append((lemma.name().replace('_', ' '), relation,
                                     related_sense.name()))
    for target, relation, related_sense in sorted(set(evidence)):
        if target == cue:
            continue
        if not graph.has_edge(cue, target):
            graph.add_edge(cue, target, evidence=[])
        graph[cue][target]['evidence'].append({
            'cue': cue, 'target': target, 'relation': relation,
            'source_sense': sense.name(), 'related_sense': related_sense,
            'definition': sense.definition(),
        })
    return graph

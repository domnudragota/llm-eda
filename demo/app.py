from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from semantic_memory.ui.snippets import SNIPPETS, snippet_source
from semantic_memory.ui.graphs import (association_subgraph, reference_subgraph, graph_figure, activation_subgraph,
                        activation_graph_figure)
from semantic_memory.integrations.lwow import (DEFAULT_REPO, load_original_functions, official_graph_path,
                         official_gender_matrix_path, load_normalized_activation)
from semantic_memory.integrations.wordnet import senses, build_reference_graph, wn

BASE = Path(__file__).resolve().parent
FEMALE_PRIMES = ['woman', 'female', 'mother', 'girl', 'feminine']
MALE_PRIMES = ['man', 'male', 'father', 'boy', 'masculine']

st.set_page_config(page_title='Code analysis', layout='wide')
st.title('The role of System 1 and System 2 semantic memory structure in human and LLM biases - code analysis')
st.caption(
    'Based on the article from Abramski et al. (2026), primarily referencing the LWOW repository, which served as a basis for the final rendition.')


@st.cache_resource(show_spinner='Calling original LWOW graph loader…')
def load_association_graph(repo_str, selected_model, graph_str=None):
    funcs = load_original_functions(Path(repo_str))
    source = Path(graph_str) if graph_str else official_graph_path(Path(repo_str), selected_model)
    columns = pd.read_csv(source, nrows=0).columns
    if not {'src', 'tgt', 'wt'} <= set(columns):
        raise ValueError('Expected an association edge-list CSV with src, tgt, wt columns, not an activation matrix.')
    graph = funcs.edgelist2graph(str(source), directed=False)
    return graph


@st.cache_data(show_spinner='Normalizing the FULL published matrix…')
def load_activation_matrix(repo_str, selected_model):
    return load_normalized_activation(Path(repo_str), selected_model)


@st.cache_data(show_spinner=False)
def load_reference_graph(cue, sense_name, layer):
    return build_reference_graph(cue, sense_name, layer)


tabs = st.tabs(['Semantic memory layers', 'Spreading activation', 'Code snippets'])
with tabs[0]:
    controls = st.columns([1, 1, 1])

    cue_count = 100
    cues = (BASE / 'data/reference/demo_cues.txt').read_text().splitlines()[:cue_count]

    association_datasets = ['Mistral', 'Llama3', 'Haiku']

    with controls[0]:
        model = st.selectbox('Free-association dataset', association_datasets)
    with controls[1]:
        cue = st.selectbox('Cue word', cues)
    with controls[2]:
        limit = st.slider('Maximum visible nodes per panel', 12, 60, 36)

    st.subheader('System 1-based semantic network layer LLM free-association layer')
    st.caption('Wrapper over LWOW repo-provided data')

    association_file = official_graph_path(DEFAULT_REPO, model)

    if not association_file.is_file():
        st.warning('Original graph missing. Run scripts/setup_lwow.ps1, then refresh.')
    else:
        try:
            graph = load_association_graph(str(DEFAULT_REPO.resolve()), model, str(association_file.resolve()))
            if cue not in graph:
                st.info('This term does not occur in the selected original graph.')
            else:
                association_columns = st.columns([1.2, 1])
                with association_columns[0]:
                    shown = association_subgraph(graph, cue, limit)
                    fig = graph_figure(shown, cue)
                    st.pyplot(fig)
                    plt.close(fig)
        except Exception as error:
            st.error(f'Could not display the graph: {error}')

    st.divider()

    st.subheader('System 2-based semantic network layers snippets')
    st.caption('Not generated from LLMs, but only snippets from WordNet.'
               ' Therefore, similar to the article\'s approach to human-related semantic layers. ')
    available_senses = []
    try:
        available_senses = senses(cue)
    except LookupError:
        st.warning('WordNet corpus missing. Run: python -m scripts.setup_wordnet. '
                   'Original association and activation views remain available.')
    if available_senses:
        definitions = {sense.name(): sense.definition() for sense in available_senses}
        parts_of_speech = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective', 's': 'Adjective', 'r': 'Adverb'}
        sense_name = st.selectbox('WordNet definitions', list(definitions),
                                  format_func=lambda
                                      name: f'Def: [{parts_of_speech[wn.synset(name).pos()]}]; {definitions[name]}')

        st.caption('Created using WordNet, as the LWOW repo targets exclusively system 1-equivalent layers.')
        for column, layer, title in zip(st.columns(2), ['definition', 'category'],
                                        ['Definitions', 'Categorical relations']):
            with column:
                st.markdown(f'**{title} layer**')
                reference = load_reference_graph(cue, sense_name, layer)
                shown = reference_subgraph(reference, cue, limit)
                fig = graph_figure(shown, cue, reference=True)
                st.pyplot(fig)
                plt.close(fig)

                if layer == 'category':
                    st.caption('Blue: synonym · rose: antonym · purple: hypernym · gold: hyponym.')
                if not reference.number_of_edges():
                    st.info('No qualifying links for this sense. Choose another sense or cue.')
    else:
        st.caption('No WordNet sense available for this cue, or corpus not installed.')

with tabs[1]:
    activation_model = st.selectbox('Spreading-activation network', ['Mistral', 'Llama3', 'Haiku'])
    st.caption('All data is precomputed, based on the original LWOW repo.')

    if not official_gender_matrix_path(DEFAULT_REPO, activation_model).is_file():
        st.warning('Activation matrix missing. Run setup script.')
    else:
        try:
            normalized = load_activation_matrix(str(DEFAULT_REPO.resolve()), activation_model)
            group = st.radio('Gender', ['Female', 'Male'], horizontal=True)
            targets = pd.read_csv(DEFAULT_REPO / 'reproducibility/data/LDT_analyses/gender_targets.csv')[group].dropna().tolist()
            available = [word for word in targets if word in normalized.index]
            selected = st.multiselect('Target words', available, default=available[:8])
            prime = st.selectbox('Prime word', FEMALE_PRIMES + MALE_PRIMES)

            if selected:
                shown = load_original_functions(DEFAULT_REPO).matricesGender(
                    normalized, selected, FEMALE_PRIMES, MALE_PRIMES)
                values = normalized[prime]
                node_limit = st.slider('Maximum visible activation nodes', 30, 100, 60, step=10)
                full_graph = load_association_graph(str(DEFAULT_REPO.resolve()), activation_model)
                displayed = activation_subgraph(full_graph, prime, selected, values, node_limit)
                st.caption('Solid links are original LWOW associations. Shortest paths and nearby words '
                           'are selected only to make a compact drawing; they are not measured activation trajectories.')
                fig = activation_graph_figure(displayed, prime, selected, values)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info('Select at least one target to visualize.')
        except Exception as error:
            st.error(f'Could not load activation output: {error}')

with tabs[2]:
    st.subheader('Code snippets')
    if (DEFAULT_REPO / 'reproducibility/FA_Functions.py').is_file():
        try:
            function_name = st.selectbox('Core functions',
                                         list(SNIPPETS),
                                         format_func=lambda name: f'{list(SNIPPETS).index(name) + 1:02d}. '
                                                                  f'{name} - {SNIPPETS[name]["origin"]}')
            st.caption(SNIPPETS[function_name]['purpose'])
            st.code(snippet_source(DEFAULT_REPO, function_name),
                    language=SNIPPETS[function_name]['language'])
        except Exception as error:
            st.warning(f'Could not display source code: {error}')

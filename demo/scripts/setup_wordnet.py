import shutil

import nltk

from semantic_memory.config import DATA_DIR

DEST = DATA_DIR / 'reference/nltk_data'


def main():
    target = DEST / 'corpora/wordnet.zip'
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        try:
            existing = nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            if not nltk.download('wordnet', download_dir=str(DEST), raise_on_error=True):
                raise RuntimeError('WordNet download failed')
        else:
            shutil.copyfile(str(existing), target)

    nltk.data.path.insert(0, str(DEST))

    from nltk.corpus import wordnet as wn
    print('WordNet', wn.get_version(), 'available at', wn.root)

if __name__ == '__main__':
    main()

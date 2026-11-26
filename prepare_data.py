"""Recreate the original catalog and chronology from Amazon Beauty 2014."""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from run_paths import CODE_DIR, DATA_ROOT, project_reference


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError: yield ast.literal_eval(line)


def metadata_categories(raw, item_map):
    categories = {int(i): None for i in item_map.values()}
    def paths(node):
        if isinstance(node, list):
            if node and all(isinstance(x, str) for x in node): yield node
            else:
                for child in node: yield from paths(child)
    for row in read_gz(raw):
        if row.get('asin') not in item_map: continue
        choices = []
        for path in paths(row.get('categories', [])):
            p = [x.strip().lower() for x in path]
            if 'beauty' in p and p.index('beauty') + 1 < len(p):
                choices.append(p[p.index('beauty') + 1])
        if choices: categories[item_map[row['asin']]] = choices[0]
    return categories


def prepare(download=False):
    reference = json.loads((CODE_DIR / 'configs/data_sources.json').read_text(encoding='utf-8'))
    raw = DATA_ROOT / 'data/raw'
    for name, url_key in [('reviews_Beauty_5.json.gz', 'reviews_url'), ('meta_Beauty.json.gz', 'metadata_url')]:
        path = raw / name
        expected = reference['files']['data/raw/' + name]['sha256']
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{project_reference(path)}: download the data first, or use --download')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + '.part')
            print(f'Downloading {reference[url_key]}', flush=True)
            with urllib.request.urlopen(reference[url_key], timeout=120) as response, temp.open('wb') as out:
                while block := response.read(1024 * 1024): out.write(block)
            if sha(temp) != expected:
                raise ValueError(f'Unexpected downloaded data hash: {project_reference(temp)}. See docs/DATA.md')
            temp.replace(path)
        if sha(path) != expected:
            raise ValueError(f'Data differs from the checkpoint reference: {project_reference(path)}. See docs/DATA.md')
    users = defaultdict(list)
    for rowno, row in enumerate(read_gz(raw / 'reviews_Beauty_5.json.gz')):
        if row.get('reviewerID') and row.get('asin') and row.get('unixReviewTime') is not None:
            users[row['reviewerID']].append((int(row['unixReviewTime']), row['asin'], rowno))
    users = {uid: sorted(rows) for uid, rows in users.items() if len(rows) >= 5}
    asins = sorted({asin for rows in users.values() for _, asin, _ in rows})
    item_map = {asin: i + 1 for i, asin in enumerate(asins)}
    cards = {}
    for row in read_gz(raw / 'meta_Beauty.json.gz'):
        asin = row.get('asin')
        if asin not in item_map: continue
        flat = []
        def visit(node):
            if isinstance(node, list):
                for child in node: visit(child)
            elif isinstance(node, str) and node.strip().lower() not in flat:
                flat.append(node.strip().lower())
        visit(row.get('categories', []))
        cats = [x for x in flat[-6:] if x != 'beauty']
        title = re.sub(r'[^a-z0-9 ]+', ' ', str(row.get('title') or asin).lower())
        cards[item_map[asin]] = dict(asin=asin, title=str(row.get('title') or asin)[:180],
            tokens=title.split()[:24], categories=cats, text=' '.join(cats + title.split()[:24]))
    for asin, iid in item_map.items():
        cards.setdefault(iid, dict(asin=asin, title=asin, tokens=[], categories=[], text=''))
    sequences = {uid: dict(items=[item_map[a] for _, a, _ in rows], times=[t for t, _, _ in rows],
        row_ids=[r for _, _, r in rows]) for uid, rows in sorted(users.items())}
    freeze = DATA_ROOT / 'freeze'
    for name, obj in [('item_map', item_map), ('cards', cards), ('sequences', sequences)]:
        path = freeze / (name + '.json')
        dump(path, obj)
        if sha(path) != reference['files']['freeze/' + path.name]['sha256']:
            raise ValueError(f'Preprocessing no longer matches the reference: {project_reference(path)}')
    categories = metadata_categories(raw / 'meta_Beauty.json.gz', item_map)
    dump(freeze / 'categories.json', categories)
    result = dict(dataset=reference['dataset'], users=len(sequences), items=len(item_map),
        interactions=sum(len(s['items']) for s in sequences.values()),
        categories=dict(Counter(c for c in categories.values() if c)),
        source_sha256={name: sha(DATA_ROOT / name) for name in reference['files']},
        categories_sha256=sha(freeze / 'categories.json'))
    dump(freeze / 'data_manifest.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    prepare(parser.parse_args().download)

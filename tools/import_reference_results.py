"""Verify archived result hashes and create a relocatable GUI viewing copy."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000001/'

def relocate(value):
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace('\\', '/')
        if PREFIX in normalized:
            return str(ROOT/(PREFIX+normalized.split(PREFIX,1)[1]))
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    reports = ROOT/'reports/2026-10-09'
    manifest = json.loads((reports/'reference_manifest.json').read_text(encoding='utf-8'))['files']
    selected = 0
    target = ROOT/PREFIX
    if target.exists() and not args.verify_only:
        raise SystemExit('Viewing copy already exists; use --verify-only. Existing results are never overwritten.')
    with zipfile.ZipFile(reports/'reference_snapshot.zip') as archive:
        if set(archive.namelist()) != set(manifest)|{'review_manifest.json'}:
            raise SystemExit('Unexpected reference archive entries')
        if json.loads(archive.read('review_manifest.json'))['files'] != manifest:
            raise SystemExit('Embedded manifest differs from reference manifest')
        for name, expected in manifest.items():
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise SystemExit(f'Archive checksum mismatch: {name}')
            if not name.startswith(PREFIX):
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name:
                raise SystemExit('Unsafe archive path')
            selected += 1
            if args.verify_only:
                continue
            destination = ROOT/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith('.json'):
                # Relocate paths only in this derived viewing copy. Frozen
                # archive remains byte-identical, including original hashes.
                data = (json.dumps(relocate(json.loads(data)), ensure_ascii=False, indent=2)+'\n').encode('utf-8')
            destination.write_bytes(data)
    if not args.verify_only:
        (target/'publication_view.json').write_text(json.dumps({
            'archival':True, 'relocated_for_viewing':True,
            'current_source_revalidated':False,
            'note':'GUI viewing copy of frozen reference results. Do not resume with current source; use a new output folder.'
        }, indent=2)+'\n', encoding='utf-8')
    print(f'Archive hashes verified: {len(manifest)}; result files: {selected}')
    if not args.verify_only:
        print('Imported GUI job c16100000001 (archival viewing copy)')

if __name__ == '__main__':
    main()

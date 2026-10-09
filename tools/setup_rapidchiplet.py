"""Fetch the pinned official analytical backend; never reset existing work."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    lock = json.loads((ROOT/'rapidchiplet.lock.json').read_text(encoding='utf-8'))
    target = ROOT/'external/rapidchiplet'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'clone', '--no-checkout', lock['repository'], str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', lock['commit']], check=True)
    actual = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'], text=True).strip()
    if actual != lock['commit']:
        raise SystemExit('Existing RapidChiplet checkout has another commit. Set RAPIDCHIPLET_ROOT explicitly or use a fresh external directory; no reset performed.')
    for name, expected in lock['reference_sha256'].items():
        data = (target/name).read_bytes()
        # Git on Windows may convert LF to CRLF. Reference files use LF.
        digest = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
        if digest != expected:
            raise SystemExit(f'Native source checksum mismatch: {name}')
    print(f'Pinned RapidChiplet verified: {lock["commit"]}\n{target}')

if __name__ == '__main__':
    main()

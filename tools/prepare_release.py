"""Build and validate only the current release; never upload anything."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
from importlib.metadata import version as installed_version
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    project = (root / 'pyproject.toml').read_text()
    version = re.search(r'^version = "([^"]+)"$', project, re.M).group(1)
    output = root / 'release' / version
    if output.exists():raise SystemExit('Release directory already exists; preserve it and use a new version.')
    if importlib.util.find_spec('twine') is None:raise SystemExit('Install build and twine before preparing a release.')
    from packaging.version import Version
    if Version(installed_version('twine')) < Version('7'):
        raise SystemExit('Install Twine >=7 for current distribution metadata support.')
    output.mkdir(parents=True)
    subprocess.run([sys.executable, '-m', 'build', '--outdir', str(output)], cwd=root, check=True)
    files = [output / f'cninvoice-{version}-py3-none-any.whl', output / f'cninvoice-{version}.tar.gz']
    for path in files:
        if not path.is_file():raise SystemExit('Expected release artifact is missing.')
    with ZipFile(files[0]) as zipped:
        names = zipped.namelist()
        metadata = BytesParser(policy=policy.default).parsebytes(zipped.read('cninvoice-'+version+'.dist-info/METADATA'))
        if metadata['Name'] != 'cninvoice' or metadata['Version'] != version:
            raise SystemExit('Distribution metadata does not match the intended release.')
        if metadata['License-Expression'] != 'MIT' or not any(name.endswith('/licenses/LICENSE') for name in names):
            raise SystemExit('MIT license metadata or license file is missing.')
        forbidden = [name for name in names if '.private/' in name or name.endswith(('.onnx', '.pt', '.yml', '.yaml', '.png', '.jpg', '.jpeg', '.pdf'))]
        if forbidden:raise SystemExit('Private data, configuration or weights found in code wheel.')
    with tarfile.open(files[1]) as source:
        if any('/.private/' in n or n.endswith(('.onnx', '.pt', '.yml', '.yaml', '.png', '.jpg', '.jpeg', '.pdf')) for n in source.getnames()):
            raise SystemExit('Private data, configuration or weights found in source archive.')
    subprocess.run([sys.executable, '-m', 'twine', 'check', '--strict', *map(str, files)], check=True)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output / 'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in checksums.items()))
    (output / 'release.json').write_text(json.dumps({'name':'cninvoice','version':version,
        'license':'MIT','files':checksums,'validation':'pass','uploaded':False}, indent=2))
    print(json.dumps({'release':str(output),'validation':'pass','uploaded':False}))


if __name__ == '__main__':main()

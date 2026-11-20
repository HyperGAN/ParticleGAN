"""Software checks for base installations and optional GIF rendering."""
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _python(source, *args):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_base_packages_import_without_plotting_dependencies():
    _python("""
        import importlib.abc
        import sys
        from typing import get_type_hints

        class NoPlotting(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'matplotlib', 'PIL'}:
                    raise AssertionError('core import requested ' + fullname)

        sys.meta_path.insert(0, NoPlotting())
        from particlegan import (
            ManipulatedPrior,
            NeuralComponentImputer,
            ParticlePrior,
            joint_reconstruction_loss,
            masked_mixture_nll,
            masked_quantile_transport_loss,
        )
        from particlegan_forge import NamedStreams, TaskSpec, render_training_gif
        from particlegan_forge.__main__ import main

        assert get_type_hints(ParticlePrior.manipulate)['return'] is ManipulatedPrior
        assert ParticlePrior(num_particles=2).sample(1)[0].shape == (1, 4)
        assert all(callable(fn) for fn in (
            joint_reconstruction_loss, masked_mixture_nll,
            masked_quantile_transport_loss, NeuralComponentImputer,
        ))
        assert NamedStreams().generator('data').device.type == 'cpu'
        assert callable(render_training_gif) and callable(main) and TaskSpec
        assert not any(name.split('.')[0] in {'matplotlib', 'PIL'} for name in sys.modules)
    """)


@pytest.mark.parametrize("missing", ["matplotlib", "PIL"])
def test_renderer_explains_missing_optional_dependency(missing, tmp_path):
    _python("""
        import importlib.abc
        from pathlib import Path
        import sys

        missing, output = sys.argv[1], Path(sys.argv[2])
        class MissingDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == missing:
                    raise ModuleNotFoundError('missing ' + fullname, name=fullname)

        sys.meta_path.insert(0, MissingDependency())
        from particlegan_forge import render_training_gif
        try:
            render_training_gif([], output, task_id='software', candidate_id='fixture', verdict='PASS')
        except ModuleNotFoundError as exc:
            assert 'particlegan[experiments]' in str(exc)
            assert exc.name.split('.')[0] == missing
        else:
            raise AssertionError('rendering without dependency succeeded')
        assert not output.exists()
    """, missing, tmp_path / "absent.gif")


def test_renderer_preserves_matplotlib_backend_and_writes_observed_frames(tmp_path):
    _python("""
        import hashlib
        from pathlib import Path
        import sys
        import matplotlib
        matplotlib.use('svg')
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        before_backend, before_figures = matplotlib.get_backend(), plt.get_fignums()
        from particlegan_forge import render_training_gif
        assert matplotlib.get_backend() == before_backend
        output = Path(sys.argv[1])
        frames = [
            {'step': 0, 'passed': False, 'metrics': {'score': 0.},
             'target_images': np.ones((1, 1, 8, 8)), 'sample_images': np.zeros((1, 1, 8, 8))},
            {'step': 1, 'passed': True, 'metrics': {'score': 1.},
             'target_images': np.ones((1, 1, 8, 8)), 'sample_images': np.ones((1, 1, 8, 8))},
        ]
        receipt = render_training_gif(frames, output, task_id='software', candidate_id='fixture', verdict='PASS')
        assert receipt['frames'] == 2
        assert receipt['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
        with Image.open(output) as image:
            assert image.format == 'GIF' and image.n_frames == 2
        assert matplotlib.get_backend() == before_backend
        assert plt.get_fignums() == before_figures
    """, tmp_path / "fixture.gif")


def test_pypi_wheel_and_sdist_remain_tiny_and_exclude_non_package_trees(tmp_path):
    _python("""
        from pathlib import Path
        import shutil
        import sys
        import tarfile
        import zipfile
        import setuptools.build_meta as build_meta

        root = Path.cwd()
        stage = Path(sys.argv[1]) / "stage"
        dist = Path(sys.argv[1]) / "dist"
        stage.mkdir()
        dist.mkdir()
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
            shutil.copy2(root / name, stage / name)
        for pkg in ("particlegan", "particlegan_forge"):
            shutil.copytree(root / pkg, stage / pkg)
        for excluded in (
            "tests", "reports", "results", "experiments", "benchmarks", "lib",
            "configs", "data", "runs", "artifacts", "docs", "examples", "ops", "scripts",
        ):
            (stage / excluded).mkdir()
            (stage / excluded / "probe.py").write_text("raise RuntimeError('should not be packaged')\\n")

        import os
        os.chdir(stage)
        wheel = dist / build_meta.build_wheel(str(dist))
        sdist = dist / build_meta.build_sdist(str(dist))

        max_bytes = 1_200_000
        assert wheel.stat().st_size < max_bytes, wheel.stat().st_size
        assert sdist.stat().st_size < max_bytes, sdist.stat().st_size

        with zipfile.ZipFile(wheel) as archive:
            wheel_roots = {name.split("/")[0] for name in archive.namelist()}
        assert all(
            r in {"particlegan", "particlegan_forge"} or r.endswith(".dist-info")
            for r in wheel_roots
        ), wheel_roots

        allowed_sdist_roots = {
            "LICENSE", "MANIFEST.in", "PKG-INFO", "README.md",
            "particlegan", "particlegan.egg-info", "particlegan_forge",
            "pyproject.toml", "setup.cfg",
        }
        with tarfile.open(sdist) as archive:
            sdist_roots = {
                parts[1] for name in archive.getnames()
                if len(parts := name.split("/")) > 1
            }
        assert sdist_roots <= allowed_sdist_roots, sdist_roots - allowed_sdist_roots
    """, tmp_path)

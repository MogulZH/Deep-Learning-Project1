"""Reproduce archived notebook variants without editing the executed notebook.

Run from the repository root after placing data.zip beside this file.
Default outputs are isolated from the original student's training directories.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
PRESETS = {
    'baseline': {'epochs': 100, 'randaugment': False, 'mixup_alpha': None},
    'randaugment': {'epochs': 200, 'randaugment': True, 'mixup_alpha': None},
    'mixup': {'epochs': 200, 'randaugment': True, 'mixup_alpha': 0.2},
}


class SetAssignments(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values
        self.changed = set()

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.values:
                node.value = ast.Constant(value=self.values[name])
                self.changed.add(name)
        return self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=PRESETS, default='mixup')
    parser.add_argument('--smoke', action='store_true', help='512 train / 256 validation, one epoch')
    parser.add_argument('--tta', action='store_true', help='Two-view horizontal-flip probability averaging at inference')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/reproduced')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    os.chdir(ROOT)
    preset = PRESETS[args.experiment]
    run_name = args.experiment + ('_smoke' if args.smoke else '')
    notebook_path = ROOT/'CIFAR10_ResNet_Project.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = {cell.id: cell.source for cell in notebook.cells if cell.cell_type == 'code'}
    run_dir = args.output_root.resolve()/run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(
        experiment=args.experiment, smoke_test=args.smoke, tta=args.tta, **preset,
        width=32, batch_size=128, seed=42,
        randaugment_num_ops=2 if preset['randaugment'] else None,
        randaugment_magnitude=9 if preset['randaugment'] else None,
        data_dir=str(args.data_dir.resolve()),
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
    )
    settings_path = run_dir/'reproduction_settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError('Settings or source changed. Use a different --output-root.')
    settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    setup = ast.parse(cells['cell-01'])
    patcher = SetAssignments({'EPOCHS': preset['epochs'], 'SMOKE_TEST': args.smoke, 'RUN_NAME': run_name})
    setup = ast.fix_missing_locations(patcher.visit(setup))
    assert patcher.changed == {'EPOCHS', 'SMOKE_TEST', 'RUN_NAME'}
    # The archived notebook's local folder locator is intentionally overridden:
    # this checkout may have any folder name, and data.zip need not be extracted yet.
    setup_source = ast.unparse(setup)
    locator_start = setup_source.index('    candidates = ')
    locator_end = setup_source.index('    sys.path.insert', locator_start)
    setup_source = setup_source[:locator_start] + '    PROJECT_DIR = Path.cwd()\n' + setup_source[locator_end:]
    anchor = 'OUT = OUTPUT_ROOT / RUN_NAME'
    assert anchor in setup_source
    setup_source = setup_source.replace(
        anchor,
        f'OUTPUT_ROOT = Path({str(args.output_root.resolve())!r})\n'
        f'DATA_DIR = Path({str(args.data_dir.resolve())!r})\n' + anchor,
        1,
    )
    namespace = {'__name__': '__main__'}
    exec(compile(setup_source, 'notebook:setup', 'exec'), namespace)
    namespace['plt'].switch_backend('Agg')
    for cell_id in ['cell-03', 'cell-05', 'cell-07']:
        code = cells[cell_id]
        if cell_id == 'cell-05' and not preset['randaugment']:
            line = '    transforms.RandAugment(num_ops=2, magnitude=9),\n'
            assert line in code
            code = code.replace(line, '', 1)
        exec(compile(code, f'notebook:{cell_id}', 'exec'), namespace)
    if preset['mixup_alpha'] is not None:
        mixup_cells = [code for code in cells.values() if code.startswith('def train_one_epoch_mixup(')]
        assert len(mixup_cells) == 1
        exec(compile(mixup_cells[0], 'notebook:mixup', 'exec'), namespace)
    training = cells['cell-09']
    if preset['mixup_alpha'] is None:
        assert 'train_one_epoch_mixup(' in training
        training = training.replace('train_one_epoch_mixup(', 'train_one_epoch(', 1)
    exec(compile(training, 'notebook:training', 'exec'), namespace)
    for cell_id in ['cell-11', 'cell-13', 'cell-15']:
        code = cells[cell_id]
        if cell_id == 'cell-13' and not args.tta:
            start = code.index('                p_original = ')
            end = code.index('            prediction_by_id.update', start)
            code = code[:start] + '                logits = model(x)\n' + code[end:]
        exec(compile(code, f'notebook:{cell_id}', 'exec'), namespace)
    print('External Kaggle accuracy is not computed locally. See results/kaggle_scores.json.')
    print('Reproduction artifacts:', run_dir)


if __name__ == '__main__':
    main()

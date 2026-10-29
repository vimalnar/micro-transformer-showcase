"""Canonical model adapter, deterministic answer-only fine-tuning, atomic persistence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = Path(os.environ.get('MICRO_TRANSFORMER_ROOT', str(ROOT.parent / 'micro-transformer'))).expanduser().resolve()
LOCK = json.loads((ROOT / 'framework-lock.json').read_text())
if not (FRAMEWORK / '.git').exists():
    raise RuntimeError('Pinned Micro-Transformer checkout missing. Run python scripts/setup_framework.py, or set MICRO_TRANSFORMER_ROOT. See README.md.')
if subprocess.check_output(['git', '-C', str(FRAMEWORK), 'rev-parse', 'HEAD'], text=True).strip() != LOCK['revision']:
    raise RuntimeError('Framework revision differs from framework-lock.json')
for name, expected in LOCK['files'].items():
    if hashlib.sha256((FRAMEWORK / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Framework source/checkpoint changed: {name}')
sys.path.insert(0, str(FRAMEWORK))
import torch
from training.harness import InferenceHarness
from training.data import encode, EOS_ID, TOKENS

torch.set_num_threads(2)
BASE = FRAMEWORK / 'baselines/v1.0.0/checkpoints/seed42-inference.pt'
ARCH = FRAMEWORK / 'models/transformer.py'


def rows(text):
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError('Practice must be text of at most 100,000 characters')
    lines = [line for line in text.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 64:
        raise ValueError('Use 1–64 examples: story and question => answer |')
    result = []
    for line in lines:
        if line.count('=>') != 1:
            raise ValueError('Each line needs exactly one =>')
        prompt, answer = map(encode, line.split('=>'))
        if EOS_ID in prompt or answer[-1] != EOS_ID or EOS_ID in answer[:-1] or len(answer) < 2:
            raise ValueError('Omit | from prompts; end each answer with one |')
        if len(prompt) + len(answer) > 129:
            raise ValueError('Example exceeds context window')
        result.append((prompt, answer))
    return result


def require_finite_logits(module, inputs, output):
    """Reject invalid numerical output before generation can turn NaNs into tokens."""
    if not torch.isfinite(output).all():
        raise ValueError('Non-finite transformer logits; no answer generated')


def validate_checkpoint_tensors(model, optimizer):
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite checkpoint tensor: '+name)
    for values in optimizer.state.values():
        for name,value in values.items():
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Non-finite checkpoint optimizer tensor: '+name)


class Session:
    def __init__(self, directory=None, scope="chatbot"):
        if scope not in ("chatbot", "devices"):
            raise ValueError("Unknown showcase")
        self.scope = scope
        self.directory = Path(directory or (os.environ.get('SHOWCASE_STATE') if scope == 'chatbot' else None) or ROOT / 'state' / scope)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pause = threading.Event()
        self.busy = False
        self.error = None
        self.harness = InferenceHarness(BASE, device='cpu', architecture=ARCH)
        self.model = self.harness.model
        self.model.requires_grad_(True)
        self.model.register_forward_hook(require_finite_logits)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.0001)
        self.steps = self.offset = 0
        self.total_loss = 0.
        self.last_loss = None
        self.recipe = None
        self.draft = {'text': (ROOT / 'data' / scope / 'default.txt').read_text(), 'rounds': 2, 'rate': .0001}
        if self.path.exists():
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
            if saved['framework'] != LOCK or saved['showcase'] != self.scope:
                raise ValueError('Saved checkpoint provenance mismatch')
            self.model.load_state_dict(saved['model'])
            self.optimizer.load_state_dict(saved['optimizer'])
            for key in ('steps', 'offset', 'total_loss', 'last_loss', 'recipe', 'draft', 'error'):
                setattr(self, key, saved[key])
            validate_checkpoint_tensors(self.model,self.optimizer)
            torch.set_rng_state(saved['rng'])

    @property
    def path(self):
        return self.directory / 'current.pt'

    def save(self):
        value = {key: getattr(self, key) for key in ('steps','offset','total_loss','last_loss','recipe','draft','error')}
        value.update(showcase=self.scope, framework=LOCK, model=self.model.state_dict(),
                     optimizer=self.optimizer.state_dict(), rng=torch.get_rng_state())
        temporary = self.path.with_suffix('.tmp')
        torch.save(value, temporary)
        temporary.replace(self.path)

    def status(self):
        with self.lock:
            return dict(showcase=self.scope, busy=self.busy, error=self.error, steps=self.steps, done=self.offset,
                        total=0 if not self.recipe else len(rows(self.recipe['text']))*self.recipe['rounds'],
                        loss=self.last_loss, mean_loss=self.total_loss/self.offset if self.offset else None,
                        draft=self.draft, recipe=self.recipe, state_sha256=self.harness.state_sha256(),
                        checkpoint_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else LOCK['files'][str(BASE.relative_to(FRAMEWORK))],
                        config=vars(self.model.config), parameters=sum(x.numel() for x in self.model.parameters()),
                        framework=LOCK, tokens=TOKENS)

    def predict(self, prompt):
        if not isinstance(prompt, str) or len(prompt) > 100000:
            raise ValueError('Prompt must be text of at most 100,000 characters')
        with self.lock:
            result = self.harness.predict(prompt)
            result.update(steps=self.steps, state_sha256=self.harness.state_sha256())
            return result

    def start(self, recipe, background=True):
        rows(recipe['text'])
        if type(recipe['rounds']) is not int or not 1 <= recipe['rounds'] <= 20 or recipe['rate'] not in (.00003,.0001,.0003):
            raise ValueError('Choose 1–20 rounds and a supported learning rate')
        with self.lock:
            if self.busy:
                raise ValueError('Training is already running')
            self.draft = dict(recipe)
            if recipe != self.recipe or self.offset >= len(rows(recipe['text']))*recipe['rounds']:
                self.recipe = dict(recipe)
                self.offset = 0
                self.total_loss = 0.
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=recipe['rate'])
            self.error = None
            self.pause.clear()
            self.busy = True
            self.save()
        if background:
            threading.Thread(target=self.run, daemon=True).start()
        else:
            self.run()

    def update(self, row):
        prompt, answer = row
        ids = prompt + answer
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(torch.tensor([ids[:-1]]))
        loss = torch.nn.functional.cross_entropy(logits[0,len(prompt)-1:], torch.tensor(answer))
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss; training stopped')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.last_loss = float(loss.detach())
        self.total_loss += self.last_loss
        self.offset += 1
        self.steps += 1

    def run(self):
        try:
            examples = rows(self.recipe['text'])
            total = len(examples)*self.recipe['rounds']
            while not self.pause.is_set():
                with self.lock:
                    if self.offset >= total:
                        break
                    self.update(examples[self.offset % len(examples)])
                    self.save()
                # Cooperatively release the GIL between completed updates so HTTP
                # status/pause handlers are not crowded out by the next update.
                time.sleep(0)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.busy = False
                self.model.eval()
                self.save()

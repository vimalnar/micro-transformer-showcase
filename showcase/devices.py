"""Python-owned bounded simulation. Assessment never supplies model tokens."""
import copy
import json
from .core import TOKENS

ADAPTER = 'device-location-messages-v1'
PLACES = ('river','lab','store','home','bridge','room','forest','dock')


def packet(place, delivered=True):
    if place not in PLACES or type(delivered) is not bool:
        raise ValueError('Choose a supported location and boolean delivery setting')
    return dict(place=place, delivered=delivered,
                text=f'hal move flag six to {place}.' +
                     (f' hal tell gia flag six at {place}.' if delivered else ''))


def prompt(events):
    if not 1 <= len(events) <= 6:
        raise ValueError('Send 1–6 events; clear the stream before a seventh event')
    return ' '.join(packet(e['place'],e['delivered'])['text'] for e in events) + ' where gia knows flag six?'


def state(events):
    return dict(actual=events[-1]['place'] if events else None,
                received=next((e['place'] for e in reversed(events) if e['delivered']), 'unknown'))


def assess(events, token_ids):
    tokens = [TOKENS[i] for i in token_ids]
    valid = len(tokens) == 2 and tokens[1] == '|' and tokens[0] in ('unknown', *PLACES)
    expected = state(events)['received']
    return dict(expected=expected, raw=' '.join(tokens), valid=valid,
                report=tokens[0] if valid else None, matches=valid and tokens[0] == expected)


def practice():
    sets = {'Delivered updates':[], 'Missing updates':[], 'Mixed sequences':[]}
    for i,a in enumerate(PLACES):
        b = PLACES[(i+3)%8]
        cases = [('Delivered updates',[packet(a),packet(b)]),
                 ('Missing updates',[packet(a),packet(b,False)]),
                 ('Mixed sequences',[packet(a,False)] if i%2 else
                  [packet(a),packet(b,False),packet(PLACES[(i+5)%8])])]
        for name, events in cases:
            sets[name].append(prompt(events)+' => '+state(events)['received']+' |')
    return sets


class Simulation:
    def __init__(self, session):
        self.session = session
        self.path = session.directory / 'simulation.json'
        self.events = []
        self.result = None
        self.error = None
        self.revision = 0
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved['adapter'] != ADAPTER:
                raise ValueError('Device simulation adapter mismatch')
            events = saved['events']
            if events:
                prompt(events)  # Validate before accepting persisted state.
            self.events = [packet(e['place'],e['delivered']) for e in events]
            self.result = saved['result']
            self.error = saved['error']
            self.revision = saved['revision']

    def save(self):
        value = dict(adapter=ADAPTER, events=self.events, result=self.result,
                     error=self.error, revision=self.revision)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,allow_nan=False))
        temporary.replace(self.path)

    def snapshot(self):
        with self.session.lock:
            current_hash = self.session.harness.state_sha256()
            return dict(adapter=ADAPTER, places=PLACES, events=copy.deepcopy(self.events),
                        revision=self.revision, **state(self.events),
                        prompt=prompt(self.events) if self.events else '',
                        result=copy.deepcopy(self.result), error=self.error,
                        stale=self.result is not None and self.result['state_sha256'] != current_hash)

    def send(self, place, delivered):
        with self.session.lock:
            events = self.events + [packet(place,delivered)]
            prompt(events)  # Seventh event is rejected without altering stream/report.
            self.events = events
            self.revision += 1
            self.result = None
            self.error = None
            self.save()  # Event survives even when inference fails.
            return self.interpret()

    def interpret(self):
        with self.session.lock:
            text = prompt(self.events)
            self.result = None
            self.error = None
            try:
                # Only encoded history reaches inference. Ground truth is assessed afterwards.
                self.result = self.session.predict(text)
                self.result['assessment'] = assess(self.events, self.result['token_ids'])
                self.result['revision'] = self.revision
            except Exception as exc:
                self.error = str(exc)
                raise
            finally:
                self.save()
            return self.snapshot()

    def clear(self):
        with self.session.lock:
            self.events = []
            self.result = None
            self.error = None
            self.revision += 1
            self.save()
            return self.snapshot()

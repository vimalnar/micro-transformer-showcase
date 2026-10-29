"""Ava: canonical components plus extension stages and a Python-owned world."""
import json

from showcase.ava.session import AvaSession, state_hash
from showcase.ava.world import ACTIONS, features
from .common import assert_equal, options


def main():
    with options(__doc__) as (args, directory):
        session = AvaSession(directory)
        before = state_hash(session.model)
        if args.train:
            # Keep rate, seed, selection and world unchanged. Extend the target
            # by one epoch so completed saved practice is not implicitly reset.
            settings = dict(session.settings, epochs=session.epoch + 1)
            session.start(settings, list(session.selection), background=False)
            if session.error:
                raise RuntimeError(session.error)
            assert state_hash(session.model) != before
        session.retry()  # Start a fresh journey without changing learned weights.
        learned = state_hash(session.model)
        trace = session.model.inspect(features(session.world), session.memory)
        print(json.dumps(dict(action=ACTIONS[trace['actionIndex']],
                              probabilities=trace['probabilities'],
                              stages=[stage['type'] for stage in trace['stages']],
                              state_sha256=learned), indent=2))
        result = session.action()  # The real model selects; Python applies rules.
        assert state_hash(session.model) == learned
        restored = AvaSession(directory)
        assert state_hash(restored.model) == learned
        assert_equal(restored.optimizer.state_dict(), session.optimizer.state_dict())
        assert restored.world == session.world and restored.memory == session.memory
        print(json.dumps(dict(tick=result['world']['tick'], reached=result['reached'],
                              decision=result['last_decision']['action']), indent=2))
        print('Model, optimizer, world and journey memory restored; action did not train.')


if __name__ == '__main__':
    main()

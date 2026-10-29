"""Chatbot: canonical inference, optional fine-tuning and exact restoration."""
import json

from showcase.core import Session
from .common import assert_equal, options

PROMPT = 'gia move lamp twelve to dock. where lamp twelve?'


def main():
    with options(__doc__) as (args, directory):
        session = Session(directory, scope='chatbot')
        before = session.harness.state_sha256()
        print('Before:', json.dumps(session.predict(PROMPT)))
        assert session.harness.state_sha256() == before
        if args.train:
            # The same answer-only gradient update used by HTML Training.
            session.start(dict(text=PROMPT + ' => dock |', rounds=1, rate=.0001), background=False)
            if session.error:
                raise RuntimeError(session.error)
            assert session.harness.state_sha256() != before
        session.save()
        restored = Session(directory, scope='chatbot')
        assert restored.harness.state_sha256() == session.harness.state_sha256()
        assert restored.steps == session.steps
        assert_equal(restored.optimizer.state_dict(), session.optimizer.state_dict())
        learned = restored.harness.state_sha256()
        print('Restored:', json.dumps(restored.predict(PROMPT)))
        assert restored.harness.state_sha256() == learned
        print('Exact weights and optimizer restored; inference left weights unchanged.')


if __name__ == '__main__':
    main()

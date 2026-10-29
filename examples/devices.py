"""Devices: Python events become controlled-language transformer inputs."""
import json

from showcase.core import Session
from showcase.devices import Simulation, practice
from .common import assert_equal, options


def main():
    with options(__doc__) as (args, directory):
        session = Session(directory, scope='devices')
        before = session.harness.state_sha256()
        if args.train:
            text = '\n'.join(practice()['Missing updates'])
            session.start(dict(text=text, rounds=1, rate=.0001), background=False)
            if session.error:
                raise RuntimeError(session.error)
            assert session.harness.state_sha256() != before
        learned = session.harness.state_sha256()
        simulation = Simulation(session)
        simulation.clear()  # Reset this example's stream, retaining its weights.
        simulation.send('river', delivered=True)
        report = simulation.send('dock', delivered=False)
        assert session.harness.state_sha256() == learned
        print(json.dumps(report, indent=2))
        # Actual location is dock; last delivered location is river. Assessment
        # checks the generated answer afterwards; it never supplies that answer.
        assert report['actual'] == 'dock' and report['received'] == 'river'
        session.save()
        restored = Session(directory, scope='devices')
        assert restored.harness.state_sha256() == session.harness.state_sha256()
        assert_equal(restored.optimizer.state_dict(), session.optimizer.state_dict())
        restored_simulation = Simulation(restored)
        assert restored_simulation.events == simulation.events
        learned = restored.harness.state_sha256()
        print('Restored report:', json.dumps(restored_simulation.interpret()['result']))
        assert restored.harness.state_sha256() == learned
        print('Model, optimizer and event stream restored; prediction did not train.')


if __name__ == '__main__':
    main()

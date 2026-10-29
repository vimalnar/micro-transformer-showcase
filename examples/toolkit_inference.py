"""The toolkit directly: no web server, application session or saved user model."""
import json

# This bootstrap verifies framework-lock.json and puts the source checkout on
# sys.path. These paths identify the canonical architecture and supplied weights.
from showcase.core import ARCH, BASE
from training.harness import InferenceHarness


def main():
    harness = InferenceHarness(BASE, device='cpu', architecture=ARCH)
    before = harness.state_sha256()
    result = harness.predict('gia move lamp twelve to dock. where lamp twelve?')
    assert harness.state_sha256() == before
    print(json.dumps(dict(prediction=result, state_sha256=before), indent=2))


if __name__ == '__main__':
    main()

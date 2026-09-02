from types import SimpleNamespace

import scripts.create_dataset as dataset_module
from scripts.create_dataset import CASES, DATASET_NAME, create_dataset


class FakeClient:
    def __init__(self) -> None:
        self.examples_call = None

    def create_dataset(self, **kwargs):
        assert kwargs == {
            "dataset_name": DATASET_NAME,
            "description": "Hybrid Agent 的路由、工具与答案回归测试",
        }
        return SimpleNamespace(id="dataset-id", name=DATASET_NAME)

    def create_examples(self, **kwargs) -> None:
        self.examples_call = kwargs


def test_create_dataset_uploads_all_versioned_cases() -> None:
    client = FakeClient()

    dataset = create_dataset(client)

    assert dataset.name == DATASET_NAME
    assert client.examples_call == {
        "dataset_id": "dataset-id",
        "examples": CASES,
    }


def test_create_dataset_loads_dotenv_before_building_client(monkeypatch) -> None:
    events = []

    class EnvironmentAwareClient(FakeClient):
        def __init__(self) -> None:
            assert events == ["dotenv-loaded"]
            super().__init__()

    monkeypatch.setattr(
        dataset_module,
        "load_dotenv",
        lambda: events.append("dotenv-loaded"),
    )
    monkeypatch.setattr(dataset_module, "Client", EnvironmentAwareClient)

    dataset_module.create_dataset()

    assert events == ["dotenv-loaded"]

#shared_data_store.py

import asyncio
from typing import Any, Dict, List, Callable, Tuple

class Observer:
    async def update(self, message: str, event_type: str, **kwargs):
        pass

class AsyncModelRetrainer(ModelRetrainer):
    async def update(self, message: str, event_type: str):
        if event_type == "dataset_update":
            # Assume `retrain_model` is an async function for retraining your model
            print(f"New data available, retraining models. Event: {message}")
            await retrain_model()

class DataPreparationTrigger(Observer):
    async def update(self, message: str, event_type: str):
        if event_type == "dataset_update":
            print(f"New dataset available, starting data preparation. Event: {message}")
            # Trigger data preparation logic here
            await prepare_data_for_model()

# Assuming `prepare_data_for_model` is an asynchronous function
async def prepare_data_for_model():
    datastore = SharedDataStore()
    dataset_name = "your_dataset"
    dataset = datastore.get_dataset(dataset_name)
    if dataset is not None:
        # Proceed with data preparation using the dataset
        ...


class ModelNotifier(Observer):
    async def update(self, message: str, event_type: str):
        if event_type == "model_update":
            print(f"New model version available. Event: {message}")
            # Add your notification logic here

class LazyDataLoader:
    def __init__(self, loader_function):
        self.loader_function = loader_function
        self.data = None

    def get_data(self):
        if self.data is None:
            self.data = self.loader_function()
        return self.data

class SharedDataStore:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.configurations: Dict[str, Any] = {}
        self.observers: List[Tuple[Observer, str, Callable, int]] = []
        self.loop = asyncio.get_event_loop()

    def set_configuration(self, name: str, value: Any):
        self.configurations[name] = value
        self.loop.create_task(self.notify_observers(f"Configuration {name} updated", "config_update"))

    def get_configuration(self, name: str, default: Any = None) -> Any:
        return self.configurations.get(name, default)

    async def notify_observers(self, message: str, event_type: str, **kwargs):
        tasks = []
        for observer, interest, condition, _ in self.observers:
            if interest == event_type and (condition is None or condition(kwargs)):
                tasks.append(observer.update(message, event_type, **kwargs))
        if tasks:
            await asyncio.gather(*tasks)

    def register_observer(self, observer: Observer, interest: str = None, condition: Callable = None, priority: int = 0):
        self.observers.append((observer, interest, condition, priority))
        self.observers.sort(key=lambda x: x[3], reverse=True)

    def add_dataset(self, name: str, dataset: Any, metadata: Dict = None, tags: List[str] = None):
        version = self.datasets.get(name, {}).get('version', 0) + 1
        self.datasets[name] = {'data': dataset, 'metadata': metadata or {}, 'version': version, 'tags': tags or []}
        self.loop.create_task(self.notify_observers(f"Dataset {name} updated to version {version}", "dataset_update"))

    def get_dataset(self, name: str, version: int = None) -> Any:
        dataset_info = self.datasets.get(name)
        if dataset_info and (version is None or dataset_info['version'] == version):
            return dataset_info['data']
        return None

    def add_model(self, name: str, model: Any, metadata: Dict = None, tags: List[str] = None):
        version = self.models.get(name, {}).get('version', 0) + 1
        self.models[name] = {'model': model, 'metadata': metadata or {}, 'version': version, 'tags': tags or []}
        self.loop.create_task(self.notify_observers(f"Model {name} updated to version {version}", "model_update"))

    def get_model(self, name: str, version: int = None) -> Any:
        model_info = self.models.get(name)
        if model_info and (version is None or model_info['version'] == version):
            return model_info['model']
        return None

    def add_dataset_lazy(self, name: str, loader_function: Callable, metadata: Dict = None, tags: List[str] = None):
        lazy_loader = LazyDataLoader(loader_function)
        self.datasets[name] = {'data': lazy_loader, 'metadata': metadata or {}, 'version': 1, 'tags': tags or []}
        self.loop.create_task(self.notify_observers(f"Lazy-loaded dataset {name} added", "dataset_update"))
       
datastore = SharedDataStore()
datastore.register_observer(AsyncModelRetrainer(), interest="dataset_update")
datastore.register_observer(ModelNotifier(), interest="model_update")
datastore.register_observer(DataPreparationTrigger(), interest="dataset_update")

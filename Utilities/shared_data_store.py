#shared_data_store.py
import asyncio

class Observer:
    async def update(self, message, event_type):
        # To be implemented by the observer, event_type can be used to differentiate the event
        # Making this method async suggests that the implementing observer might perform async operations
        pass
    
class ModelRetrainer(Observer):
    async def update(self, message, event_type):
        if event_type == "dataset_update":
            print(f"New data available, retraining models. Event: {message}")
            # Trigger model retraining here

class ModelNotifier(Observer):
    async def update(self, message, event_type):
        if event_type == "model_update":
            print(f"New model version available. Event: {message}")
            # Send notifications or perform other actions here

class SharedDataStore:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.observers = []
        self.loop = asyncio.get_event_loop()

    async def notify_observers(self, message, event_type):
        tasks = []
        for observer, interest in self.observers:
            if interest is None or interest == event_type:
                if asyncio.iscoroutinefunction(observer.update):
                    tasks.append(observer.update(message, event_type))
                else:
                    # Wrap the synchronous observer update method in a coroutine
                    tasks.append(self.loop.run_in_executor(None, observer.update, message, event_type))
        if tasks:
            await asyncio.gather(*tasks)

    def register_observer(self, observer, interest=None):
        self.observers.append((observer, interest))

    def add_dataset(self, name, dataset, metadata=None, tags=None):
        version = self.datasets.get(name, {}).get('version', 0) + 1
        self.datasets[name] = {'data': dataset, 'metadata': metadata or {}, 'version': version, 'tags': tags or []}
        asyncio.run_coroutine_threadsafe(self.notify_observers(f"Dataset {name} updated to version {version}", "dataset_update"), self.loop)

    def get_dataset(self, name, version=None):
        dataset_info = self.datasets.get(name)
        if dataset_info and (version is None or dataset_info['version'] == version):
            return dataset_info['data']
        return None

    def add_model(self, name, model, metadata=None, tags=None):
        version = self.models.get(name, {}).get('version', 0) + 1
        self.models[name] = {'model': model, 'metadata': metadata or {}, 'version': version, 'tags': tags or []}
        asyncio.run_coroutine_threadsafe(self.notify_observers(f"Model {name} updated to version {version}", "model_update"), self.loop)

    def get_model(self, name, version=None):
        model_info = self.models.get(name)
        if model_info and (version is None or model_info['version'] == version):
            return model_info['model']
        return None
    
    def register_observer(self, observer, interest=None, condition=None):
        self.observers.append((observer, interest, condition))

    async def notify_observers(self, message, event_type, **kwargs):
        tasks = []
        for observer, interest, condition in self.observers:
            if interest == event_type and (condition is None or condition(kwargs)):
                if asyncio.iscoroutinefunction(observer.update):
                    tasks.append(observer.update(message, event_type))
                else:
                    tasks.append(self.loop.run_in_executor(None, observer.update, message, event_type))
        await asyncio.gather(*tasks)

    def set_configuration(self, name, value):
        """Set a configuration parameter."""
        self.configurations[name] = value

    def get_configuration(self, name, default=None):
        """Get a configuration parameter, returning a default if not found."""
        return self.configurations.get(name, default)
           
class LazyDataLoader:
    def __init__(self, loader_function):
        self.loader_function = loader_function
        self.data = None

    def get_data(self):
        if self.data is None:
            self.data = self.loader_function()
        return self.data

# Usage of LazyDataLoader in SharedDataStore
def add_dataset_lazy(self, name, loader_function, metadata=None, tags=None):
    lazy_loader = LazyDataLoader(loader_function)
    self.datasets[name] = {'data': lazy_loader, 'metadata': metadata or {}, 'version': 1, 'tags': tags or []}
    # Notify observers about the new lazy-loaded dataset
    asyncio.run_coroutine_threadsafe(self.notify_observers(f"Lazy-loaded dataset {name} added", "dataset_update"), self.loop)


"""
Base class for NLP processing pipelines
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Callable

class Pipeline(ABC):
    """
    Abstract base class for NLP pipelines.
    
    Pipelines define a sequence of processing steps (processors)
    to be applied to input data.
    """
    
    def __init__(self, processors: List[Callable] = None):
        """
        Initialize the pipeline.
        
        Args:
            processors (List[Callable], optional): 
                A list of processor functions or objects with a __call__ method. 
                Defaults to None, subclasses should define default processors.
        """
        self.processors = processors if processors is not None else self._get_default_processors()

    @abstractmethod
    def _get_default_processors(self) -> List[Callable]:
        """
        Subclasses must implement this to return a default list of processors.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: Any, **kwargs) -> Any:
        """
        Process the input data through the pipeline.
        
        Args:
            data (Any): The input data to process.
            **kwargs: Additional keyword arguments for processors.
            
        Returns:
            Any: The processed data.
        """
        raise NotImplementedError

    def add_processor(self, processor: Callable, index: int = -1):
        """
        Add a processor to the pipeline at a specific index.
        
        Args:
            processor (Callable): The processor function or object.
            index (int, optional): Position to insert the processor. 
                                   Defaults to -1 (append).
        """
        if index == -1:
            self.processors.append(processor)
        else:
            self.processors.insert(index, processor)

    def remove_processor(self, processor_name: str):
        """
        Remove a processor by its name (if it has a __name__ or name attribute).
        
        Args:
            processor_name (str): The name of the processor to remove.
        """
        self.processors = [
            p for p in self.processors 
            if getattr(p, '__name__', getattr(p, 'name', None)) != processor_name
        ]

    def get_processor_names(self) -> List[str]:
        """
        Get the names of the processors in the pipeline.
        """
        names = []
        for p in self.processors:
            name = getattr(p, '__name__', getattr(p, 'name', str(p)))
            names.append(name)
        return names

"""
Advanced Distributed Processing System for ThaiNLP
"""
from typing import Dict, List, Any, Optional, Callable, Union
import os
import time
import json
import threading
import queue
import multiprocessing as mp
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from ..scaling.auto_scaler import ServiceMetrics

@dataclass
class TaskMetadata:
    """Task metadata for tracking"""
    task_id: str
    status: str
    start_time: float
    end_time: Optional[float] = None
    error: Optional[str] = None
    result: Any = None

class DataPartitioner:
    """Intelligent data partitioning for distributed processing"""
    
    def __init__(self, strategy: str = "dynamic"):
        """Initialize partitioner
        
        Args:
            strategy: Partitioning strategy ('static', 'dynamic', or 'adaptive')
        """
        self.strategy = strategy
        self.partition_stats = {}
        
    def partition_data(
        self,
        data: Union[List[Any], pd.DataFrame],
        num_partitions: int
    ) -> List[Any]:
        """Partition data for distributed processing
        
        Args:
            data: Data to partition
            num_partitions: Number of partitions
            
        Returns:
            List of data partitions
        """
        if isinstance(data, pd.DataFrame):
            return self._partition_dataframe(data, num_partitions)
        return self._partition_list(data, num_partitions)
        
    def _partition_list(
        self,
        data: List[Any],
        num_partitions: int
    ) -> List[List[Any]]:
        """Partition list data
        
        Args:
            data: List to partition
            num_partitions: Number of partitions
            
        Returns:
            List of partitioned lists
        """
        if self.strategy == "static":
            # Simple equal-sized partitions
            partition_size = len(data) // num_partitions
            return [
                data[i:i + partition_size]
                for i in range(0, len(data), partition_size)
            ]
            
        elif self.strategy == "dynamic":
            # Balance partitions based on data size
            total_size = sum(len(str(item)) for item in data)
            target_size = total_size // num_partitions
            
            partitions = []
            current_partition = []
            current_size = 0
            
            for item in data:
                item_size = len(str(item))
                if current_size + item_size > target_size and current_partition:
                    partitions.append(current_partition)
                    current_partition = []
                    current_size = 0
                    
                current_partition.append(item)
                current_size += item_size
                
            if current_partition:
                partitions.append(current_partition)
                
            return partitions
            
        else:  # adaptive
            # Use historical statistics to optimize partitioning
            if not self.partition_stats:
                return self._partition_list(data, num_partitions)
                
            # Calculate optimal partition sizes based on processing history
            total_items = len(data)
            partition_sizes = []
            
            for i in range(num_partitions):
                if str(i) in self.partition_stats:
                    stats = self.partition_stats[str(i)]
                    relative_speed = stats['processed_items'] / stats['processing_time']
                    partition_sizes.append(relative_speed)
                else:
                    partition_sizes.append(1.0)
                    
            # Normalize sizes
            total_size = sum(partition_sizes)
            partition_sizes = [s / total_size for s in partition_sizes]
            
            # Create partitions
            partitions = []
            start_idx = 0
            
            for size in partition_sizes:
                end_idx = start_idx + int(size * total_items)
                partitions.append(data[start_idx:end_idx])
                start_idx = end_idx
                
            return partitions
            
    def _partition_dataframe(
        self,
        df: pd.DataFrame,
        num_partitions: int
    ) -> List[pd.DataFrame]:
        """Partition DataFrame
        
        Args:
            df: DataFrame to partition
            num_partitions: Number of partitions
            
        Returns:
            List of partitioned DataFrames
        """
        if self.strategy == "static":
            return np.array_split(df, num_partitions)
            
        elif self.strategy == "dynamic":
            # Balance partitions based on memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            target_size = memory_usage // num_partitions
            
            partitions = []
            current_partition = pd.DataFrame()
            
            for _, row in df.iterrows():
                row_size = row.memory_usage(deep=True)
                
                if current_partition.memory_usage(deep=True).sum() + row_size > target_size and not current_partition.empty:
                    partitions.append(current_partition)
                    current_partition = pd.DataFrame(columns=df.columns)
                    
                current_partition = pd.concat([current_partition, row.to_frame().T])
                
            if not current_partition.empty:
                partitions.append(current_partition)
                
            return partitions
            
        else:  # adaptive
            if not self.partition_stats:
                return np.array_split(df, num_partitions)
                
            # Use historical statistics for optimal partitioning
            total_rows = len(df)
            partition_sizes = []
            
            for i in range(num_partitions):
                if str(i) in self.partition_stats:
                    stats = self.partition_stats[str(i)]
                    relative_speed = stats['processed_items'] / stats['processing_time']
                    partition_sizes.append(relative_speed)
                else:
                    partition_sizes.append(1.0)
                    
            # Normalize sizes
            total_size = sum(partition_sizes)
            partition_sizes = [s / total_size for s in partition_sizes]
            
            # Create partitions
            partitions = []
            start_idx = 0
            
            for size in partition_sizes:
                end_idx = start_idx + int(size * total_rows)
                partitions.append(df.iloc[start_idx:end_idx])
                start_idx = end_idx
                
            return partitions
            
    def update_stats(
        self,
        partition_id: str,
        processed_items: int,
        processing_time: float
    ):
        """Update partition statistics
        
        Args:
            partition_id: Partition identifier
            processed_items: Number of items processed
            processing_time: Processing time in seconds
        """
        if partition_id not in self.partition_stats:
            self.partition_stats[partition_id] = {
                'processed_items': 0,
                'processing_time': 0
            }
            
        stats = self.partition_stats[partition_id]
        stats['processed_items'] += processed_items
        stats['processing_time'] += processing_time

class DistributedProcessor:
    """Advanced distributed processing system"""
    
    def __init__(
        self,
        num_workers: int = None,
        partitioner: Optional[DataPartitioner] = None
    ):
        """Initialize distributed processor
        
        Args:
            num_workers: Number of worker processes (default: CPU count)
            partitioner: Data partitioner instance
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.partitioner = partitioner or DataPartitioner()
        
        # Task tracking
        self.tasks = {}
        self.task_queue = queue.PriorityQueue()
        self._lock = threading.Lock()
        
        # Initialize worker pool
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_tasks,
            daemon=True
        )
        self._monitor_thread.start()
        
    def process_data(
        self,
        data: Union[List[Any], pd.DataFrame],
        process_fn: Callable,
        combine_fn: Optional[Callable] = None,
        priority: int = 1,
        **kwargs
    ) -> Any:
        """Process data in distributed manner
        
        Args:
            data: Data to process
            process_fn: Function to apply to each partition
            combine_fn: Function to combine results (default: list)
            priority: Task priority (lower is higher)
            **kwargs: Additional arguments for process_fn
            
        Returns:
            Combined processing results
        """
        # Partition data
        partitions = self.partitioner.partition_data(data, self.num_workers)
        
        # Create task metadata
        task_id = str(time.time())
        self.tasks[task_id] = TaskMetadata(
            task_id=task_id,
            status="pending",
            start_time=time.time()
        )
        
        # Submit partitions for processing
        futures = []
        
        for i, partition in enumerate(partitions):
            future = self.executor.submit(
                self._process_partition,
                partition,
                process_fn,
                i,
                task_id,
                **kwargs
            )
            futures.append(future)
            
        # Wait for results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.tasks[task_id].error = str(e)
                self.tasks[task_id].status = "failed"
                raise
                
        # Combine results
        if combine_fn:
            final_result = combine_fn(results)
        else:
            final_result = results
            
        # Update task metadata
        self.tasks[task_id].status = "completed"
        self.tasks[task_id].end_time = time.time()
        self.tasks[task_id].result = final_result
        
        return final_result
        
    def _process_partition(
        self,
        partition: Any,
        process_fn: Callable,
        partition_id: int,
        task_id: str,
        **kwargs
    ) -> Any:
        """Process single data partition
        
        Args:
            partition: Data partition
            process_fn: Processing function
            partition_id: Partition identifier
            task_id: Task identifier
            **kwargs: Additional arguments
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            result = process_fn(partition, **kwargs)
            
            # Update partition statistics
            if isinstance(partition, (list, pd.DataFrame)):
                processed_items = len(partition)
            else:
                processed_items = 1
                
            self.partitioner.update_stats(
                str(partition_id),
                processed_items,
                time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            self.tasks[task_id].error = str(e)
            self.tasks[task_id].status = "failed"
            raise
            
    def _monitor_tasks(self):
        """Background task monitoring thread"""
        while True:
            with self._lock:
                # Clean up old completed tasks
                current_time = time.time()
                for task_id in list(self.tasks.keys()):
                    task = self.tasks[task_id]
                    if task.status in ["completed", "failed"]:
                        if current_time - task.end_time > 3600:  # Keep for 1 hour
                            del self.tasks[task_id]
                            
            time.sleep(60)  # Check every minute
            
    def get_task_status(self, task_id: str) -> Optional[TaskMetadata]:
        """Get task status
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task metadata if found
        """
        return self.tasks.get(task_id)
        
    def get_active_tasks(self) -> List[TaskMetadata]:
        """Get all active tasks
        
        Returns:
            List of active task metadata
        """
        return [
            task for task in self.tasks.values()
            if task.status not in ["completed", "failed"]
        ]
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel task if possible
        
        Args:
            task_id: Task identifier
            
        Returns:
            Whether task was cancelled
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == "pending":
                task.status = "cancelled"
                task.end_time = time.time()
                return True
        return False 
from abc import ABC, abstractmethod
from typing import (
    Any, 
    Dict, 
    List, 
    Union, 
    Optional,
)

# Note: This base interface will evolve as additional memory systems are integrated.
# As new methods and capabilities are standardized across implementations, this class
# may be updated to ensure a consistent API across different memory backends.

class BaseMemoryLayer(ABC):
    """
    Abstract base class for memory layers that defines a unified interface for various memory 
    algorithms. This class follows the template method pattern and provides common methods 
    that should be implemented by concrete memory layer classes.
    
    The interface is designed to be compatible with popular memory frameworks like Mem0, 
    A-MEM, LangMem, and other memory systems, providing a consistent API for memory 
    operations across different implementations.
    """

    @abstractmethod
    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """
        Add a single message to the memory layer.

        Parameters
        ----------
        message : Dict[str, str]
            A message dictionary containing 'role' and 'content' keys.
        **kwargs : Any
            Additional keyword arguments that may be required by specific implementations.
            For example, temporal systems may require a 'timestamp'.

        Returns
        -------
        None
            This method doesn't return a value but stores the message in the memory system.
        """
        pass

    @abstractmethod
    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        """
        Add a list of messages to the memory layer.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            A list of message dictionaries, each containing 'role' and 'content' keys.
            The 'role' typically indicates the speaker (e.g., 'user', 'assistant', 'system'),
            and 'content' contains the actual message text.
        **kwargs : Any
            Additional keyword arguments that may be required by specific implementations.
            For example, 'timestamp' may be required for temporal memory systems.

        Returns
        -------
        None
            This method doesn't return a value but stores the messages in the memory system.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 10, **kwargs) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        Retrieve memory entries based on a query string.

        Parameters
        ----------
        query : str
            The search query to find relevant memories.
        k : int, default 10
            Maximum number of memories to return.
        **kwargs : Any
            Additional keyword arguments for retrieval customization, such as filters,
            search parameters, or user-specific settings.

        Returns
        -------
        memories : List[Dict[str, Union[str, Dict[str, Any]]]]
            A list of memory entries matching the query. Each entry typically contains:
            - 'content': str - The actual memory content
            - 'metadata': Dict[str, Any] - Associated metadata including timestamps,
              keywords, categories, and other relevant information.
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory entry.

        Parameters
        ----------
        memory_id : str
            The unique identifier of the memory entry to delete.

        Returns
        -------
        success : bool
            True if the memory was successfully deleted, False otherwise.
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, **kwargs) -> bool:
        """
        Update an existing memory entry.

        Parameters
        ----------
        memory_id : str
            The unique identifier of the memory entry to update.
        **kwargs : Any
            Keyword arguments containing the fields to update. The specific fields
            depend on the memory system implementation (e.g., content, keywords,
            tags, metadata, etc.).

        Returns
        -------
        success : bool
            True if the memory was successfully updated, False otherwise.
        """
        pass

    @abstractmethod
    def save_memory(self) -> None:
        """
        Save the memory state to the implementation-configured storage location.
        Implementations may persist both configuration and serialized memory data.

        Returns
        -------
        None
            This method doesn't return a value but persists the memory state to disk.
        """
        pass

    @abstractmethod
    def load_memory(self, user_id: Optional[str] = None) -> bool:
        """
        Load the memory state for a specific user.

        Parameters
        ----------
        user_id : str, optional, default None
            The identifier of the user whose memory state should be loaded. If not
            provided, the implementation may use an internal default.

        Returns
        -------
        success : bool
            True if the user's memory state was successfully loaded, False otherwise.
        """
        pass
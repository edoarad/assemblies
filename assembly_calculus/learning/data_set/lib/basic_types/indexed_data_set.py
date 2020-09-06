from abc import ABCMeta, abstractmethod

from assembly_calculus.learning.data_set import DataPoint
from assembly_calculus.learning.data_set import DataSetBase


class IndexedDataSet(DataSetBase, metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the data_set set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    This class contains any and all shared logic between different types of DataSets.
    """
    @abstractmethod
    def _get_item(self, item) -> DataPoint:
        """
        Get the element in the <item> index in the data_set.
        :return: int
        """
        pass

    def __getitem__(self, item) -> DataPoint:
        if not 0 <= item < 2 ** self.input_size:
            raise IndexError(f"Item of index {item} is out of range (choose an"
                             f"index between 0 and {2 ** self.input_size})")

        return self._process_output_values(self._get_item(item))

Usage example for multithreaded: 
The following is code for a multithreaded function that sums a list of numbers.

```python
import numpy as np
from brain.performance.multithreaded import multithreaded

@multithreaded
def sum_list(list_chunk):
    return sum(list_chunk)


@sum_list.set_params
def sum_list_params(thread_count, lst):
    n = int(np.ceil(len(lst) / thread_count))
    # Return the list of args, kwargs to give to each thread.
    return [((lst[n * i:n * i + n],), {}) for i in range(thread_count)]


@sum_list.set_after
def sum_list_after(sums):
    return sum([s or 0 for s in sums])


print(sum_list([1, 2, 3, 4, 5]))
```

Will result in:
```
>>> 15
```

## Introduction
This implementation of assembly calculus has been done by talented students in Tel Aviv University as their coursework in "Assembly Calculus Workshop". 
The details of the workshop are in [this document](https://docs.google.com/document/d/1czzQee_afvhgzptOJJFCVQ1OuU_OoXoYw1m_3ew0zCM/edit#).


## TODO
1. Delete this
6. clean up code
7. fix problems in high plasticity (sometimes, assembly tests fail)

## Installation

1. Clone the repository and enter it:

    ```sh
    $ git clone https://github.com/edoarad/assemblies
    ```

2. In a Linux system, run the installation script and activate the virtual environment:
    ```sh
    $ ./scripts/install.sh
    ...
    $ source .env/bin/activate
    [Brain] $ # you're good to go!
    ```
   The script will also attempt to run the simulations.
   After these commands, one can also run the simulations using their name without a path, such as `merge_simulation`
   instead of `simulations.assemblies.merge_simulation`, like so:
   ```sh
   $ merge_simulation
   ...
   $ simplified_simulation
   ```

   For Windows users, install the necessary libraries with:
   ```sh
   > pip install -r requirements.txt
   ```
   and run the simulations using the full directory. For example:
   ```sh
   > python -m simulations.assemblies.merge_simulation
   ```

3. To check that everything is working as expected, run the tests:

    ```sh
    $ pytest tests/
    ...
    ```

    If there are memory errors, 

## Usage

The `Brain` packages provides the following classes:
    
- `Area`

    This class represents an Area in the brain.

    ```pycon
    >>> from assembly_calculus.brain import Area
    >>> area = Area(n = 1000, k = 31, beta = 0.05)
    ```

    Where as in the paper:
    * `n` is the number of neurons in each area, 
    * `k` is the number of "winners": neurons that fire after each round (which by default are calculated as `sqrt(n)`), 
    * and `beta` is the plasticity parameter for synapses going _into_ `area`. 

    The area class provides methods for handling areas within the brain.
    - `read`
        returns the most active assembly in an area. Can be called by an area object in the following manner:
        ```pycon
        >>> read_assembly = area.read(preserve_brain=True, brain=brain)
        ```
    
- `Connectome`
    
    A connectome represents the graph of connections between areas and stimuli of the brain.

    A connectome is initialized with a list of areas and stimuli,
    and `p` = the probability of an existing synaptic connection between any 2 neurons.

    For example:

    ```python
    from brain.connectome import Connectome
    from brain.components import Area, Stimulus
    def simple_conn():
        a = Area(n=1000, k=31, beta=0.05)
        b = Area(n=1000, k=31, beta=0.05)
        s = Stimulus(n=1, beta=0.05)
        return Connectome(p=0.3, areas=[a,b], stimuli=[s]), a, b, s
    ```

    One may also add areas and or stimuli after creating a connectome, like so:

    ```pycon
    >>> conn.add_area(a)
    >>> conn.add_stimulus(s)
    ```

    After initializing a connectome,
    one may fire the connectome from given sources to given destinations.

    For example:

    ```pycon
    >>> conn, a, b, s = simple_conn()
    >>> conn.fire({s: [a], a: [b]})
    ```

    This will cause a projection from `s` to `a` and from `a` to `b`.
    
- `Brain`
        
- `Assembly`
    
    This class represents an Assembly in the brain.
    
    ```pycon
    >>> from assembly_calculus import Area, Stimulus, Assembly, BrainRecipe
    >>> area = Area(n = 1000, k = 31, beta = 0.1)
    >>> stim = Stimulus(31)
    >>> assembly = Assembly([stim], area)
    >>> recipe = BrainRecipe(stim, area, assembly)
    ```
    
    `BrainRecipe` is used to automatically generate a brain equipped with a working connectome and any relation that has been defined thus far such as `project` between assemblies that has been defined before activating the recipe. 
    
    Representing multiple assemblies and operating on them can be achieved using `|` in the following manner:
    
    ```pycon
    >>> # This represents the set of these 'ass1', 'ass2', 'ass3':
    >>> assembly_set = (ass1 | ass2 | ass3)         
    >>> # Using `Ellipsis`, this represents the singleton containing 'ass':
    >>> assembly_singelton = (ass| ...)             
    ```
    The assembly class provides methods for manipulating assemblies within the brain.
    
    - `project`
        takes an assembly in a certain area and creates a copy of that assembly
        in another area. Can be called using `>>` in the following manner:
    
        ```pycon
        >>> with recipe:
                assembly_ac = assembly_a >> area_c  # assuming assembly_a is in area_a, this creates a projection in area_c
                assembly_bc = assembly_b >> area_c
        ```
       
    - `merge`
        takes two assemblies from two areas and creates an assembly in a third area
        that fires together with both of them. Can be called using `|` and `>>` in the following manner:
        
        ```pycon
        >>> with recipe:
                (assembly_a | assembly_b) >> area_c
        ```
        
    - `associate`
        takes two assemblies in a certain area and associates them such that
        they fire together. Can be called on more than one assembly using `|` in the following manner:
        
        ```pycon
        >>> with recipe:
                (assembly_ac | ...).associate(assembly_bc | ...)
        ```
        



- `Learning`

    This module is used for assessing the use of the brain architecture as a framework for statistical learning tasks. Documentation is [here](assembly_calculus/brain/performance/multithreaded/README.md)

## General notes

The current (Sep 20) implementation has large memory requirements. Depending on the computer performing the simulation, the brain should be no more than `n=1000` or perhaps `n=10000`.
To fix this, it may be better to maintain in memory only visited synapses or only the exponent of `(1+beta)` which can later be substituted into a polynomial.


In order to handle possible memory issues, it's recommended to increase the swap region in your hard drive.
Links:
https://bogdancornianu.com/change-swap-size-in-ubuntu/
https://docs.alfresco.com/3.4/tasks/swap-space-lin.html

[ON WINDOWS COMPUTERS] In the computer's search bar, search for 'View Advanced System Properties'.
In that window, under 'Performance', press 'Settings...'. On the opened window, switch to 'Advanced', and under 'Virtual Memory',
press 'Change' to change the swapping region.

### Multithreading
See [documentation here](assembly_calculus/brain/performance/multithreaded/README.md)
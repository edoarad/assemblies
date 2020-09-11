![build status](https://travis-ci.org/BrainProjectTau/Brain.svg?branch=master)
[![coverage](https://img.shields.io/badge/coverage-404-lightgrey)](https://img.shields.io/badge/coverage-404-lightgrey)
[![codesize](https://img.shields.io/github/languages/code-size/Assemblies-Performance/assemblies)](https://img.shields.io/github/languages/code-size/Assemblies-Performance/assemblies)
[![laziness](https://img.shields.io/badge/laziness-0-brightgreen)](https://img.shields.io/badge/laziness-0-brightgreen)
[![performance](https://img.shields.io/badge/%D7%A9%D7%A0%D7%9E%D7%A8%D7%95%D7%A4%D7%A8%D7%A2%D7%A4-100%25-ff69b4)](https://img.shields.io/badge/%D7%A9%D7%A0%D7%9E%D7%A8%D7%95%D7%A4%D7%A8%D7%A2%D7%A4-100%25-ff69b4)
[![assemblies](https://img.shields.io/badge/assemblies-0-red)](https://img.shields.io/badge/assemblies-0-red)
[![guyde](https://img.shields.io/badge/guyde-100%25-9cf)](https://github.com/guyde2011)
[![badges](https://img.shields.io/badge/badges-118%25-ffcc99)](https://img.shields.io/badge/badges-118%25-ffcc99)
[![meta](https://img.shields.io/badge/meta-1000000000000000000000%25-80dfff)](https://img.shields.io/badge/meta-1000000000000000000000%25-80dfff)
[![bananas](https://img.shields.io/badge/bananas-0-ffdb4d)](https://www.youtube.com/watch?v=aKn0HddzuWM)
[![exbananas](https://img.shields.io/badge/exbananas-1-yellow)](https://www.youtube.com/watch?v=vnciwwsvNcc)


## TODO
1. Delete this
2. link to other readmes
3. remove useless branches
4. explain memory issues and set TODOs for ideas on fixing these
5. go over installation and use instructions
6. clean up code
7. fix problems in high plasticity

## Installation

1. Clone the repository and enter it:

    ```sh
    $ git clone git@github.com:BrainProjectTau/Brain.git
    ...
    $ cd Brain/
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

In order to handle possible memory issues, it's recommended to increase the swap region in your hard drive.
Links:
https://bogdancornianu.com/change-swap-size-in-ubuntu/
https://docs.alfresco.com/3.4/tasks/swap-space-lin.html

[ON WINDOWS COMPUTERS] In the computer's search bar, search for 'View Advanced System Properties'.
In that window, under 'Performance', press 'Settings...'. On the opened window, switch to 'Advanced', and under 'Virtual Memory',
press 'Change' to change the swapping region.

### Multithreading
See [documentation here](assembly_calculus/brain/performance/multithreaded/README.md)
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
    >>> from Brain import Area
    >>> area = Area(beta = 0.1, n = 1000, k = 31)
    ```

    The area class provides methods for handling areas within the brain.
    - `read`
        returns the most active assembly in an area. Can be called by an area object in the following manner:
        ```pycon
        >>> read_assembly = area.read(preserve_brain=True, brain=brain)
        ```
        
- `Assembly`
    
    This class represents an Assembly in the brain.
    
    ```pycon
    >>> from assembly_calculus import Area, Stimulus, Assembly, BrainRecipe
    >>> area = Area(beta = 0.1, n = 1000, k = 31)
    >>> stim = Stimulus(1000 ** 0.5)
    >>> assembly = Assembly([stim], area)
    >>> recipe = BrainRecipe(stim, area, assembly)
    ```
    
    Representing multiple assemblies and operating on them can be achieved using `|` in the following manner:
    
    ```pycon
    >>> assembly_set = (ass1 | ass2 | ass3)         # this represents the set of these 'ass1', 'ass2', 'ass3'
    >>> assembly_singelton = (ass| ...)             # this represents the singleton containing 'ass'
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
        
    
- `Connectome`
    
    Sub-package which holds the structre of the brain.
    The sub-package defines the following classes:
    
    - `Connectome`
        Abstract class which defines the API which a general connectome should have.
        This class should be inhereted and implemented.
        
        ```pycon
        >>> from Connectome import Connectome
        >>> class LazyConnectome(Connectome):
        >>>     #implementation of a specific connectome
        >>>> connectome = LazyConnectome()
        >>> area = Area(beta = 0.1, n = 1000, k = 31)
        >>> connectome.add_area(area)
        ```
    - `NonLazyRandomConnectome` 
        Already implemented Connectome which by decides it's edge by chance.
        This Connectome doesn't use any kind of laziness.
       
       ```pycon
        >>> from Connectome import NonLazyRandomConnectome
        >>>> connectome = NonLazyRandomConnectome()
        >>> area = Area(beta = 0.1, n = 1000, k = 31)
        >>> connectome.add_area(area)
        ```
    - `To be continued`
        More ways to implement a connectome can be applied simply by inhereting from Connectome and implementing it's API.
    
- `Brain`

    This class represents a simulated brain, with it's connectome which holds the areas, stimuli, and all the synapse weights.

    ```pycon
    >>> from Brain import Brain, NonLazyRandomConnectome, Area
    >>> connectome = NonLazyRandomConnectome()
    >>> area = Area(beta = 0.1, n = 1000, k = 31)
    >>> connectome.add_area(area)
    >>> brain = Brain(connectome)
    ```

## General notes

In order to handle possible memory issues, it's recommended to increase the swap region in your hard drive.
[ON WINDOWS COMPUTERS] In the computer's search bar, search for 'View Advanced System Properties'.
In that window, under 'Performance', press 'Settings...'. On the opened window, switch to 'Advanced', and under 'Virtual Memory',
press 'Change' to change the swapping region.
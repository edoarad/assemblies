![build status](https://travis-ci.org/BrainProjectTau/Brain.svg?branch=master)
![coverage](https://codecov.io/gh/BrainProjectTau/Brain/master/graph/badge.svg)
## Installation

1. Clone the repository and enter it:

    ```sh
    $ git clone git@github.com:edoarad/assemblies.git
    ...
    $ cd assemblies/
    ```

2. Run the installation script and activate the virtual environment:

    ```sh
    $ ./scripts/install.sh
    ...
    $ source .env/bin/activate
    [assemblies] $ # you're good to go!
    ```

3. To check that everything is working as expected, run the tests:


    ```sh
    $ pytest tests/
    ...
    ```

## Usage

The `assemblies` packages provides the following classes:
    
- `Area`

    This class represents an Area in the brain.

    ```pycon
    >>> from assemblies import Area
    >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
    ```

- `Assembly`
    
    This class represents an Assembly in the brain.
    
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
        >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
        >>> connectome.add_area(area)
        ```
    - `NonLazyRandomConnectome` 
        Already implemented Connectome which by decides it's edge by chance.
        This Connectome doesn't use any kind of laziness.
       
       ```pycon
        >>> from Connectome import NonLazyRandomConnectome
        >>>> connectome = NonLazyRandomConnectome()
        >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
        >>> connectome.add_area(area)
        ```
    - `To be continued`
        More ways to implement a connectome can be applied simply by inhereting from Connectome and implementing it's API.
    
- `Brain`

    This class represents a simulated brain, with it's connectome which holds the areas, stimuli, and all the synapse weights.

    ```pycon
    >>> from assemblies import Brain, NonLazyRandomConnectome, Area
    >>> connectome = NonLazyRandomConnectome()
    >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
    >>> connectome.add_area(area)
    >>> brain = Brain(connectome)
    ```

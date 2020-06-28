# Heurisp
*(What is it)* A repository of **Python2** implemented **H**euristics for **S**ampling-based **P**ath (Motion) Planner **(HSP)**. 

Currently, it includes these heuristics:

1. Orientation-Aware Space Exploration (OSE, referred to [^1]). It is one of the state-of-the-art heuristics according to [^2].



## How to use

- **Orientation-Aware Space Exploration** (Details included in ```test/test.py```)

```python
from heurisp import OrientationSpaceExplorer as OSExplorer
# see test directory for details to set arguments.
explorer = OSExplorer()
explorer.initialize(start, goal, grid_map, grid_res, grid_ori)
explorer.exploring()
```


## How to install

- **PyPI**

```shell script
$ pip2 install heurisp
```
- **From source**

```shell script
$ git clone https://github.com/liespace/pyHSP.git
$ cd pyHSP
$ python setup.py sdist
# install
$ pip install heurisp -f dist/* --no-cache-dir
# or upload yours
# $ twine upload dist/*
```



## Reference

[^1]: Chen, Chao, Markus Rickert, and Alois Knoll. "Path planning with orientation-aware space exploration guided heuristic search for autonomous parking and maneuvering." 2015 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2015.

[^2]: Banzhaf, Holger, et al. "Learning to predict ego-vehicle poses for sampling-based nonholonomic motion planning." IEEE Robotics and Automation Letters 4.2 (2019): 1053-1060.
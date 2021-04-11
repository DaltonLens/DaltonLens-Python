# Guidelines to work with notebooks and git

Install a tool like [nbstripout](https://github.com/kynan/nbstripout) to avoid merge conflicts on the output and avoid making the files very large on git by stripping the cells output:

```
pip3 install nbstripout
nbstripout --install --global
```

# Maintaining

This maintaining guide is for the central maintainer of the `DeepRetail` project. 


## Workflow

Maintain contribute checks:

1. Forked version: Does it build? Does it install?
2. Check code edits
3. Merge Pull Requests
4. Git Pull
5. Make edits, version in readmy.md, setup.py and version.py
6. Upgrade if needed: 
```sh
py -m pip install --upgrade pip
py -m pip install --upgrade twine
```
7. Check if the package builds
```sh
py -m build
```
8. Check if the package installs locally
```sh
py -m pip install .
```
9. Delete old tar/whl files in dist folder of old verions
10. Upload to test pypi
```sh
py -m twine upload --repository testpypi dist/*
```
11. Apply git Stash: pypirc file
12. Upload to pypi
```sh
py -m twine upload dist/* --config-file .pypirc
```
13. Save git Stash: pypirc file
14. Commit and push to git
15. Notify active contributers of new version release





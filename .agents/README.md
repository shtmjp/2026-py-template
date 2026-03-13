# Repo-local agent assets

This repository keeps reusable skills in the `https://github.com/shtmjp/my-skills.git` submodule at `.agents/my-skills`.

## Fresh clone

Initialize submodules with one of the following workflows:

```bash
git clone --recurse-submodules <repo-url>
```

or

```bash
git submodule update --init --recursive
```

## Update to the latest skills

To pull the latest `main` from the upstream skills repository into this repo:

```bash
git submodule update --init --remote .agents/my-skills
git add .agents/my-skills
git commit -m "Update my-skills"
```

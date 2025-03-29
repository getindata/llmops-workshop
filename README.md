# llmops-workshop

### Software
* Python >=3.11 (for Colab compatibility use Python 3.11.x)
* [pyenv](https://github.com/pyenv/pyenv)
* [Poetry](https://python-poetry.org/docs/)
* [Ollama](https://ollama.com/)


```bash
PYTHON_VERSION=3.11.11
pyenv install $PYTHON_VERSION
pyenv local $PYTHON_VERSION
poetry env use $PYTHON_VERSION
poetry update
```
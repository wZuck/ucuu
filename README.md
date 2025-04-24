# You Can You Up (UCUU) 🚀
--------

[![Tests](https://github.com/wZuck/ucuu/actions/workflows/python-app.yml/badge.svg)](https://github.com/wZuck/ucuu/actions/workflows/python-app.yml) [![GitHub Pages](https://github.com/wZuck/ucuu/actions/workflows/gh-pages.yml/badge.svg?branch=master)](https://github.com/wZuck/ucuu/actions/workflows/gh-pages.yml) [![PyPI Package](https://github.com/wZuck/ucuu/actions/workflows/publish.yml/badge.svg)](https://github.com/wZuck/ucuu/actions/workflows/publish.yml)
> **⚠️ Important:**  
> The majority of the code in this repository is generated using AI coding tools such as GitHub Copilot (GPT-4o) and TRAE (Doubao 1.5 Pro).  

## 1. Brief Introduction

**UCUU** is a Python utility library for function wrapping and proxying. It supports adding proxy logic to functions via decorators or patching, making it easy to extend, debug, and test.

---

## 2. Install and Usage ⚙️

### 2.1 Installation

You can install UCUU from PyPI:

```bash
pip install ucuu
```

Or, clone this repository and install locally:

```bash
git clone https://github.com/wZuck/ucuu.git
cd ucuu
pip install .
```

### 2.2 Usage

#### 2.2.1 Decorator mode (wrap for functions)

> Example: Use a decorator to add a proxy function.  
> **Effect:** When `my_func` is called, it prints "My function logic" and then the proxy prints "Hello, hello ucuu".

```python
from ucuu.decorator import ucuu

@ucuu("package_utils.print_ucuu_hello", ending_words="hello ucuu")
def my_func():
    print("My function logic")
```

#### 2.2.2 Patch mode (wrap for external functions)

> Example: Patch an existing function with a proxy.  
> **Effect:** When `some_func` is called, it prints "Original logic" and then the proxy prints "Hello, patch mode".

```python
from ucuu.decorator import ucuu

def some_func():
    print("Original logic")

some_func = ucuu("package_utils.print_ucuu_hello", ending_words="patch mode")(some_func)
```

#### 2.2.3 Register proxy functions

> Example: Implement a proxy function to be called by the decorator or patch.  
> **Effect:** Prints different messages depending on the `ending_words` argument.

```python
# tests/package_utils/test_print.py
def print_ucuu_hello(ending_words=None, *args, **kwargs):
    if ending_words is None:
        print("No Ending Words.")
    elif ending_words == "please raise errors":
        raise NotImplementedError('Raise Error due to requests')
    else:
        print(f"Hello, {ending_words}")
```

---

## 3. Demos in Testcases 🧪

- See the `tests/` directory for test cases covering decorator usage, patching, exception handling, argument binding, and more.


---

## 4. License 📄

[License](./LICENSE)

---

## 5. Contribute 🤝

Contributions via PR or issues are welcome!  
For suggestions or questions, please leave a message at [GitHub Issues](https://github.com/wZuck/ucuu/issues).

---


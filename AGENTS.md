## Python 代码风格

新增或修改 Python 业务代码时，函数签名必须尽量补全类型注解，包括参数类型和返回值类型。

推荐：

```python
def add(a: int, b: int) -> int:
    return a + b
```

避免
```python
def add(a, b):
    return a + b
```

如果类型暂时无法精确表达，可以使用 Any，但不要无理由省略类型注解
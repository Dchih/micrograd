# micrograd
a neural network demo

## 绘制和导出计算图

`draw_dot` 是 `micrograd/draw.py` 中的函数。`engine.py` 已引入它，支持直接运行文件或通过模块运行；导入 `Value` 时不会执行示例或生成文件。

绘图需要 Python 的 `graphviz` 包和系统的 Graphviz 程序。在 macOS 上安装：

```sh
brew install graphviz
.venv/bin/python -m pip install graphviz
```

在项目根目录运行（使用已有的虚拟环境）：

```sh
.venv/bin/python micrograd/engine.py
# 也可以使用：.venv/bin/python -m micrograd.engine
```

示例会执行反向传播，并导出到 `outputs/computation_graph.svg`。SVG 可以用浏览器打开并放大查看。

在自己的代码或项目根目录的 Jupyter Notebook 中：

```python
from micrograd.engine import Value
from micrograd.draw import draw_dot

a = Value(2.0, label='a')
b = Value(3.0, label='b')
out = a * b
out.label = 'out'
out.backward()

dot = draw_dot(out)
dot.render(filename='my_graph', directory='outputs', format='svg', cleanup=True)
# 导出 PNG 或 PDF：把 format='svg' 改成 'png' 或 'pdf'。
# filename 不需要包含扩展名，Graphviz 会自动添加。
```

`draw_dot(out)` 只创建图对象；调用 `.render(...)` 才会导出文件。`cleanup=True` 会删除中间 DOT 源文件，保留最终图片。Notebook 中可以把 `dot` 放在单元格最后一行直接显示，或使用 `display(dot)`。

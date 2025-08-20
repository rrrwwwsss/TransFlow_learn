# Fashion-MNIST 基本图像分类训练文档

## 1. 数据加载与预处理

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 可视化前25张训练图片
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

**解释**:

- `fashion_mnist.load_data()` 自动加载 Fashion-MNIST 数据集，如果本地没有，会从缓存或指定目录读取。
- 将像素值归一化到 `[0, 1]`。
- 可视化部分训练图片检查数据是否正常。

---

## 2. 模型定义

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

**解释**:

- `Sequential` 表示顺序模型。
- `Flatten` 将 28x28 灰度图摊平为 784 向量。
- `Dense(128, activation='relu')` 是隐藏层，有128个神经元，ReLU激活。
- `Dense(10)` 是输出层，10类分类，输出 logits。

---

## 3. 编译与训练

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=10)
```

**解释**:

- `optimizer='adam'`：自适应优化器。
- `loss=SparseCategoricalCrossentropy(from_logits=True)`：多分类损失，标签为整数。
- `metrics=['accuracy']`：训练过程中显示准确率。
- `fit` 训练模型 10 个 epoch。

---

## 4. 添加 Softmax 层用于预测概率

```python
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
```

**解释**:

- 将模型输出 logits 转换为概率分布。
- `probability_model.predict(test_images)` 可直接得到各类别概率。

---

## 5. 模型保存

```python
# 保存 Keras 原生格式
probability_model.save("my_model.keras")

# 加载
new_model = tf.keras.models.load_model("my_model.keras")
```

**解释**:

- 保存整个模型，包括结构、权重和优化器状态。
- `.keras` 是推荐扩展名。
- 之后可直接加载继续训练或做推理。

---

## 6. 测试模型与预测

```python
# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\n测试集准确率: {:.2f}%".format(test_acc * 100))

# 使用概率模型预测前10张图像
predictions = probability_model.predict(test_images[:10])
for i, probs in enumerate(predictions):
    predicted_label = np.argmax(probs)
    print(f"第{i+1}张图像预测类别: {class_names[predicted_label]}, 概率: {probs[predicted_label]:.4f}")
```

**解释**:

- `model.evaluate` 在测试集上计算损失和准确率。
- `probability_model.predict` 输出各类别概率。
- `np.argmax` 找到概率最大的类别作为预测结果。

---

## 7. 整体训练与测试流程总结

1. **加载数据集** → 归一化 → 可视化。
2. **定义模型结构** → Flatten + Dense + Dense。
3. **编译模型** → 设定优化器、损失函数和指标。
4. **训练模型** → `fit` 进行多轮迭代。
5. **建立概率模型** → 在 logits 后加 Softmax 方便预测概率。
6. **保存模型** → Keras 原生格式或 HDF5 文件，便于后续加载和使用。
7. **测试与预测** → 使用测试集评估准确率，并对单张或多张图像输出预测类别和概率。


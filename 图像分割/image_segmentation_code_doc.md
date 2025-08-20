# 图像分割任务代码说明文档

本文档对上传的 TensorFlow 图像分割代码进行逐步说明，涵盖了 **数据加载、数据增强、模型搭建、训练与可视化、类别加权处理** 等部分。

---

## 1. 导入依赖包

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
```

- `tensorflow`：核心深度学习框架。
- `tensorflow_datasets`：提供标准数据集（这里用 `oxford_iiit_pet`）。
- `pix2pix`：提供 U-Net 训练所需的上采样工具。
- `matplotlib`：用于结果可视化。

---

## 2. 数据集加载与预处理

```python
dataset, info = tfds.load(
    'oxford_iiit_pet:4.*.*',
    with_info=True,
    data_dir='../data/my_tfds_cache'
)
```

- `oxford_iiit_pet`：宠物图像分割数据集。
- `with_info=True`：返回 `info`，包含类别数、样本数等信息。
- `data_dir`：指定数据缓存路径。

### 数据归一化与调整大小

```python
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask
```

- **图像归一化**：像素值缩放到 `[0,1]`。
- **掩膜标签处理**：减去 1，使类别从 `0` 开始。
- **尺寸调整**：统一到 `128×128`。

### 数据管道构建

```python
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
```

- `AUTOTUNE`：自动优化数据加载并行度。
- `BATCH_SIZE=64`：一次训练的样本数。

### 数据增强

```python
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
```

- 数据增强：**水平翻转**，增加多样性，减少过拟合。

---

## 3. 数据批处理

```python
train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)
```

- `.cache()`：缓存数据，加快训练。
- `.shuffle()`：打乱数据顺序。
- `.batch()`：按批次训练。
- `.prefetch()`：异步预取数据，加速训练。

---

## 4. 模型构建：U-Net

编码器：使用 **MobileNetV2** 提取特征。

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False
```

解码器：Pix2Pix 上采样。

```python
up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]
```

U-Net 主体：

```python
def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3,
                                           strides=2, padding='same')
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
```

---

## 5. 模型编译与可视化

```python
OUTPUT_CLASSES = 3
model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
tf.keras.utils.plot_model(model, show_shapes=True)
```

- 优化器：Adam。
- 损失函数：逐像素交叉熵。

---

## 6. 模型预测与展示

```python
def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
```

预测可视化：

```python
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
```

训练过程回调：

```python
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
```

---

## 7. 模型训练

```python
EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])
```

---

## 8. 训练结果可视化

```python
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
```

---

## 9. 类别权重与样本加权

### 错误示范：class_weight

```python
try:
    model_history = model.fit(train_batches, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              class_weight={0:2.0, 1:2.0, 2:1.0})
    assert False
except Exception as e:
    print(f"Expected {type(e).__name__}: {e}")
```

### 正确做法：逐像素 sample_weight

```python
def add_sample_weights(image, label):
    class_weights = tf.constant([2.0, 2.0, 1.0])
    class_weights = class_weights/tf.reduce_sum(class_weights)
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return image, label, sample_weights

weighted_model = unet_model(OUTPUT_CLASSES)
weighted_model.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

weighted_model.fit(train_batches.map(add_sample_weights),
                   epochs=1,
                   steps_per_epoch=10)
```

---

# 总结

1. 本代码实现了 **U-Net 图像分割**。  
2. 数据加载、预处理、增强全部基于 `tf.data` 流水线。  
3. 使用 **MobileNetV2 编码器 + Pix2Pix 解码器** 搭建 U-Net。  
4. 可视化预测效果，实时观察训练进度。  
5. 针对类别不平衡，演示了 **逐像素 sample_weight** 的解决方案。

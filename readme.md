## 🇬🇧 English Version

This repository contains a modular implementation of a Deep Learning and Convolutional Neural Network (CNN) framework built entirely from scratch using only **NumPy**. The primary objective of this project is to understand deep learning from first principles by manually implementing every stage of the training pipeline, including convolution, pooling, backpropagation, optimization algorithms, weight initialization strategies, and parameter updates without relying on automatic differentiation or high-level frameworks such as TensorFlow or PyTorch.

The framework is designed around a dynamic architecture, allowing users to build custom CNN models by stacking an arbitrary number of **Convolutional**, **Pooling**, and **Fully Connected** layers while maintaining a clean and modular object-oriented implementation.

### 🚀 Key Features

* **Dynamic CNN Architecture:** Easily construct custom CNN models by combining any number of Convolutional, Pooling, and Fully Connected layers through the flexible `Neural Networks` class.
* **Dual Convolution Engine:** Implements both traditional **Vanilla 2D Convolution** and an optimized **im2col-based vectorized convolution** for efficient large-scale matrix computation.
* **Analytical Convolution Backpropagation:** Complete manual derivation and implementation of gradients with respect to input feature maps, convolution kernels, and bias parameters without automatic differentiation.
* **Comprehensive Pooling Layers:** Full implementations of **Max Pooling** and **Average Pooling**, including mathematically correct backward propagation algorithms.
* **Custom Padding Operations:** Configurable Zero-Padding implementation supporting different convolution configurations.
* **Advanced Weight Initialization:** Supports both **Xavier (Glorot)** and **He (Kaiming)** initialization strategies for stable gradient propagation during deep network training.
* **Vectorized Activation Functions:** Efficient implementations of **ReLU** and **Sigmoid**, together with their analytical derivatives for backpropagation.
* **Multiple Optimization Algorithms:** From-scratch implementations of **Momentum**, **RMSprop**, and **Adam** optimizers.
* **Mini-batch Gradient Descent:** Efficient mini-batch training pipeline with configurable batch sizes for scalable CNN optimization.
* **Cross-Entropy Loss:** Complete implementation of multi-class Cross-Entropy Loss using vectorized NumPy operations.
* **Complete End-to-End Training Pipeline:** Covers data preprocessing, forward propagation, backward propagation, parameter initialization, optimization, and model training entirely from scratch.

---

### 📊 Empirical Analysis & Experimental Results

The framework was experimentally validated on a subset of **1,500 MNIST samples** to verify the mathematical correctness of every implemented component.

During training, several important deep learning properties were successfully confirmed:

> 💡 **Gradient Flow Verification:** Forward propagation, convolution backpropagation, pooling backpropagation, and optimizer updates consistently preserved tensor dimensions while producing mathematically correct gradients throughout the entire computational graph.

> 📉 **Successful Optimization:** The manually implemented optimization algorithms (Adam, RMSprop, and Momentum) consistently reduced the Cross-Entropy Loss during training, confirming the correctness of the analytical derivatives and parameter update rules.

> ⚙️ **Stable Training Initialization:** Both Xavier and He initialization strategies provided stable gradient propagation and accelerated convergence during network optimization.

---

### 📂 Architecture & Core Components

The entire implementation is organized inside the main Jupyter Notebook (`.ipynb`) and is divided into independent modules:

* **Data Pipeline:** Dataset loading, preprocessing, visualization, reshaping, and Zero-Center normalization.
* **Core Mathematical Operations:** ReLU, Sigmoid, Zero Padding, Vanilla Convolution, Vectorized Convolution (`im2col`), and complete Convolution Backpropagation.
* **Pooling Modules:** Forward and Backward implementations of Max Pooling and Average Pooling.
* **Training Components:** Xavier Initialization, He Initialization, Cross-Entropy Loss, Optimizer class (Momentum, RMSprop, Adam), Mini-batch training pipeline, and the dynamic `Neural Networks` class responsible for model construction and end-to-end training.

---

## 🗺️ Future Roadmap

- [ ] Batch Normalization
- [ ] Dropout Regularization
- [ ] Softmax Layer
- [ ] Additional Learning Rate Schedulers
- [ ] Data Augmentation Pipeline
- [ ] Model Serialization (Save / Load)
- [ ] GPU Acceleration (CuPy)
- [ ] Additional CNN Architectures (LeNet, Resnet) with Pytorch

---
## 🇮🇷 نسخه فارسی

<div dir="rtl">

این مخزن شامل یک پیاده‌سازی مدولار از یک فریمورک یادگیری عمیق و شبکه عصبی کانولوشنی (CNN) است که **به‌صورت کامل از پایه (From Scratch) و تنها با استفاده از کتابخانه NumPy** توسعه یافته است. هدف اصلی این پروژه، درک عمیق مفاهیم یادگیری عمیق از اصول اولیه از طریق پیاده‌سازی دستی تمامی مراحل آموزش شبکه، شامل کانولوشن، پولینگ، پس‌انتشار خطا، روش‌های مقداردهی اولیه وزن‌ها، الگوریتم‌های بهینه‌سازی و به‌روزرسانی پارامترها، بدون استفاده از کتابخانه‌هایی مانند TensorFlow یا PyTorch است.

این فریمورک بر پایه یک معماری پویا طراحی شده است و به کاربر اجازه می‌دهد با ترکیب هر تعداد دلخواه از **لایه‌های Convolution، Pooling و Fully Connected**، شبکه‌های CNN سفارشی خود را به‌سادگی ایجاد و آموزش دهد.

### 🚀 قابلیت‌های کلیدی

* **معماری پویای CNN:** امکان ساخت شبکه‌های CNN سفارشی با هر تعداد دلخواه از لایه‌های کانولوشن، پولینگ و Fully Connected از طریق کلاس `Neural Networks`.
* **موتور دوگانه کانولوشن:** پیاده‌سازی هر دو روش **Vanilla Convolution** و نسخه بهینه‌شده مبتنی بر **im2col** برای افزایش سرعت محاسبات ماتریسی.
* **پس‌انتشار کامل کانولوشن:** محاسبه تحلیلی گرادیان نسبت به ورودی، فیلترها و بایاس‌ها بدون استفاده از قابلیت Auto-Differentiation.
* **مجموعه کامل لایه‌های Pooling:** پیاده‌سازی کامل **Max Pooling** و **Average Pooling** به همراه الگوریتم‌های دقیق Backward Propagation.
* **عملیات Padding:** پیاده‌سازی Zero Padding با اندازه‌های قابل تنظیم برای پشتیبانی از ساختارهای مختلف کانولوشن.
* **مقداردهی اولیه پیشرفته:** پیاده‌سازی روش‌های **Xavier (Glorot)** و **He (Kaiming)** جهت حفظ پایداری انتشار گرادیان در شبکه‌های عمیق.
* **توابع فعال‌سازی برداری:** پیاده‌سازی کامل توابع **ReLU** و **Sigmoid** به همراه مشتق تحلیلی آن‌ها برای فاز Backpropagation.
* **بهینه‌سازهای پیشرفته:** پیاده‌سازی کامل الگوریتم‌های **Momentum**، **RMSprop** و **Adam** از پایه.
* **آموزش Mini-batch:** پیاده‌سازی کامل آموزش مبتنی بر Mini-batch با اندازه‌های قابل تنظیم برای افزایش کارایی فرآیند آموزش.
* **تابع هزینه Cross-Entropy:** پیاده‌سازی برداری تابع هزینه چندکلاسه Cross-Entropy تنها با استفاده از NumPy.
* **خط لوله کامل آموزش:** شامل پیش‌پردازش داده‌ها، انتشار رو به جلو (Forward)، پس‌انتشار خطا (Backward)، مقداردهی اولیه پارامترها، بهینه‌سازی و آموزش کامل شبکه از ابتدا.

---

### 📊 تحلیل تجربی و نتایج آموزش

به‌منظور ارزیابی صحت تمامی اجزای پیاده‌سازی‌شده، این فریمورک روی زیرمجموعه‌ای شامل **۱۵۰۰ نمونه از دیتاست MNIST** مورد آزمایش قرار گرفت.

در طول آموزش، ویژگی‌های مهم زیر با موفقیت تأیید شدند:

> 💡 **صحت انتشار گرادیان:** عملیات Forward، Backward کانولوشن، Backward پولینگ و به‌روزرسانی پارامترها در تمام لایه‌ها، ابعاد تنسورها را به‌درستی حفظ کرده و گرادیان‌های کاملاً سازگار با روابط ریاضی تولید کردند.

> 📉 **بهینه‌سازی موفق شبکه:** پیاده‌سازی اختصاصی بهینه‌سازهای Adam، RMSprop و Momentum توانست مقدار تابع هزینه Cross-Entropy را به‌صورت پایدار در طول آموزش کاهش دهد که نشان‌دهنده صحت مشتق‌گیری‌های تحلیلی و فرآیند به‌روزرسانی پارامترها است.

> ⚙️ **پایداری آموزش:** استفاده از روش‌های مقداردهی اولیه Xavier و He موجب انتشار پایدار گرادیان‌ها و همگرایی سریع‌تر شبکه در مراحل آموزش شد.

---

### 📂 ساختار اجزای اصلی پروژه

منطق ریاضی و ساختار عملیاتی پروژه به‌صورت ماژولار در فایل اصلی Jupyter Notebook (`.ipynb`) سازماندهی شده است:

* **خط لوله داده:** بارگذاری دیتاست، نمایش نمونه‌ها، پیش‌پردازش، تغییر ابعاد داده‌ها و نرمال‌سازی Zero-Center.
* **عملیات پایه:** پیاده‌سازی ReLU، Sigmoid، Zero Padding، Vanilla Convolution، کانولوشن مبتنی بر `im2col` و الگوریتم کامل Backpropagation کانولوشن.
* **لایه‌های Pooling:** پیاده‌سازی Forward و Backward برای Max Pooling و Average Pooling.
* **اجزای آموزش:** مقداردهی اولیه Xavier و He، تابع هزینه Cross-Entropy، کلاس Optimizer (شامل Momentum، RMSprop و Adam)، آموزش Mini-batch و کلاس پویای `Neural Networks` برای ساخت، مدیریت و آموزش کامل شبکه.

---

## 🗺️ نقشه راه آینده

- [ ] **Batch Normalization:** پیاده‌سازی Batch Normalization از پایه جهت افزایش پایداری و سرعت همگرایی شبکه.
- [ ] **Dropout Regularization:** افزودن لایه Dropout برای کاهش بیش‌برازش و افزایش قدرت تعمیم.
- [ ] **Softmax Layer:** پیاده‌سازی مستقل لایه Softmax برای دسته‌بندی چندکلاسه.
- [ ] **Learning Rate Schedulers:** اضافه کردن زمان‌بندی‌های مختلف نرخ یادگیری.
- [ ] **Data Augmentation:** پیاده‌سازی تکنیک‌های افزایش داده برای مسائل بینایی ماشین.
- [ ] **Model Serialization:** امکان ذخیره و بارگذاری مدل‌های آموزش‌دیده.
- [ ] **GPU Acceleration (CuPy):** استفاده از CuPy جهت افزایش سرعت محاسبات روی GPU.
- [ ] **Additional CNN Architectures:** پیاده‌سازی معماری‌های کلاسیک مانند LeNet و Resnet از پایه با Pytorch .

</div>
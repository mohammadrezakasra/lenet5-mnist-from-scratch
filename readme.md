<div align="center">

# 🧠 Autonomous CNN & Deep Learning Framework from Scratch

### چارچوب پیشرفته و ماژولار شبکه عصبی کانولوشنی از ابتدا با نام‌پای

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-darkgreen.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Framework](https://img.shields.io/badge/Framework-From--Scratch-orange.svg?style=flat-square)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**Language / زبان**

**[🇬🇧 English Version](#-english-version)** | **[🇮🇷 نسخه فارسی](#-نسخه-فارسی)**

</div>

---

# 🇬🇧 English Version

This repository features a production-grade, modular Deep Learning and Convolutional Neural Network (CNN) framework engineered **completely from scratch using only NumPy**. Designed around a dynamic, scalable architecture, it allows developers to assemble custom deep topologies by combining an arbitrary sequence of convolutional, pooling, and dense layers.

Every single mathematical layer, analytical derivative, and adaptive optimization mechanics has been implemented from first principles—bypassing high-level deep learning engines like PyTorch or TensorFlow to achieve full control over the tensor graph.

---

## 🚀 Core Architectural Matrix

| Component | Implemented Technologies & Methods |
| :--- | :--- |
| **Convolution Engine** | Dual Mode: Vanilla Spatial 2D Conv & High-Performance Vectorized im2col Engine |
| **Pooling Suite** | Spatial Max Pooling & Average Pooling with exact Analytical Gradient Trackers |
| **Optimizers** | Standalone Class featuring Adam, RMSprop, and Momentum from scratch |
| **Initializers** | He (Kaiming) & Xavier (Glorot) distributions for gradient stabilization |
| **Activations & Loss** | Vectorized ReLU, Sigmoid functions, and Multi-class Categorical Cross-Entropy |
| **Infrastructure** | Dynamic Topology orchestrator (Neural Networks), Customizable Zero-Padding, Mini-batch SGD |

---

## 📊 Empirical Analysis & Experimental Verification

The framework's mathematical accuracy and backpropagation mechanics were thoroughly stress-tested using a localized subset of **1,500 MNIST** handwritten digits.

- **Determinism & Dimensional Sanity:** Tensor dimensions and spatial channels remained strictly invariant across nested Forward → Backward → Parameter Update iterations.
- **Loss Optimization Convergence:** Cross-Entropy loss combined with Adam and RMSprop demonstrated stable monotonic convergence.
- **Initialization Benchmarking:** Both He and Xavier initialization successfully prevented vanishing/exploding gradients.

---

## 📂 File Structure & Execution Map

```text
├── Data Pipeline
│   ├── Dataset Loading
│   ├── 4D Reshaping
│   ├── Visualizers
│   └── Zero-Center Preprocessing
│
├── Core Spatial Ops
│   ├── ReLU
│   ├── Sigmoid
│   ├── Padding
│   ├── Vanilla Convolution
│   ├── Vectorized Convolution (im2col)
│   └── Convolution Backward
│
├── Pooling Modules
│   ├── Max Pooling
│   ├── Average Pooling
│   ├── max_pool_backward
│   └── avg_pool_backward
│
└── Training Backbone
    ├── He & Xavier Initialization
    ├── Cross-Entropy Loss
    ├── Optimizer Class
    └── Neural Networks Class
```

---

## 🗺️ Future Roadmap

- [ ] Batch Normalization
- [ ] Dropout Regularization
- [ ] Standalone Softmax Layer
- [ ] Learning Rate Schedulers (StepLR, Cosine Annealing)
- [ ] Data Augmentation
- [ ] Model Serialization (.pkl)
- [ ] GPU Acceleration using CuPy
- [ ] Classical Architectures (LeNet, ResNet) implemented with PyTorch

---

# 🇮🇷 نسخه فارسی

این مخزن شامل یک فریمورک پیشرفته، مدولار و کاملاً شیءگرا برای یادگیری عمیق و شبکه‌های عصبی کانولوشنی (CNN) است که **تنها با استفاده از کتابخانه NumPy** و بدون اتکا به فریمورک‌هایی مانند PyTorch یا TensorFlow توسعه یافته است.

معماری این پروژه به‌صورت پویا طراحی شده و امکان ساخت شبکه‌های عصبی دلخواه از ترکیب لایه‌های کانولوشن، پولینگ و لایه‌های تمام‌متصل را فراهم می‌کند.

تمامی روابط ریاضی، مشتقات تحلیلی، الگوریتم‌های پس‌انتشار خطا و بهینه‌سازهای تطبیقی از پایه و بدون استفاده از کتابخانه‌های یادگیری عمیق پیاده‌سازی شده‌اند.

---

## 🚀 ماتریس ساختاری و قابلیت‌های کلیدی

| بخش | قابلیت‌ها |
| :--- | :--- |
| **هسته کانولوشن** | کانولوشن دوبعدی ساده (Vanilla) + نسخه برداری‌شده مبتنی بر im2col |
| **لایه‌های پولینگ** | پیاده‌سازی کامل Max Pooling و Average Pooling همراه با Backward دقیق |
| **بهینه‌سازها** | Adam، RMSprop و Momentum از پایه |
| **مقداردهی اولیه** | He (Kaiming) و Xavier (Glorot) |
| **توابع فعال‌سازی و هزینه** | ReLU، Sigmoid و Cross-Entropy |
| **زیرساخت آموزش** | معماری پویا، Padding سفارشی و Mini-batch SGD |

---

## 📊 تحلیل تجربی و نتایج آموزش

صحت روابط ریاضی و مکانیزم Backpropagation با استفاده از **۱۵۰۰ نمونه** از دیتاست MNIST مورد ارزیابی قرار گرفت.

- **پایداری ابعاد تنسورها** در تمام مراحل Forward و Backward حفظ شد.
- **کاهش یکنواخت تابع هزینه** با استفاده از Adam و RMSprop مشاهده شد.
- **مقداردهی اولیه He و Xavier** از محوشدگی و انفجار گرادیان جلوگیری کرد.

---

## 📂 ساختار پروژه

```text
├── خط لوله داده
│   ├── بارگذاری دیتاست
│   ├── تبدیل به تنسور چهار بعدی
│   ├── نمایش تصاویر
│   └── نرمال‌سازی Zero-Center
│
├── عملیات پایه
│   ├── ReLU
│   ├── Sigmoid
│   ├── Padding
│   ├── کانولوشن ساده
│   ├── کانولوشن برداری‌شده (im2col)
│   └── Backward کانولوشن
│
├── لایه‌های پولینگ
│   ├── Max Pooling
│   ├── Average Pooling
│   ├── max_pool_backward
│   └── avg_pool_backward
│
└── موتور آموزش
    ├── مقداردهی اولیه
    ├── Cross-Entropy
    ├── کلاس Optimizer
    └── کلاس Neural Networks
```

---

## 🗺️ نقشه راه آینده

- [ ] Batch Normalization
- [ ] Dropout Regularization
- [ ] لایه مستقل Softmax
- [ ] زمان‌بندی نرخ یادگیری (StepLR و Cosine Annealing)
- [ ] Data Augmentation
- [ ] ذخیره و بارگذاری مدل (.pkl)
- [ ] شتاب‌دهی GPU با استفاده از CuPy
- [ ] پیاده‌سازی معماری‌های LeNet و ResNet با PyTorch

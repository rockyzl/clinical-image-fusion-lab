# 先用大白话理解这个项目

这个项目问的是：**同样一张皮肤照片，如果再给模型年龄、病变部位、大小和症状，它能不能分得更准？**

“多模态”在这里就是两种信息：照片和表格。照片来自实际公开研究数据；表格是与这张照片对应的记录，不是另外编出来或从其他患者那里拼过来的。

## 在网页上怎么操作

1. 选一个公开样本。左侧是模型实际看到的照片裁剪，右侧是同一条记录的临床资料。
2. 点“让模型计算”。第一次会下载模型文件，之后在当前浏览器里计算，没有付费模型 API 调用。
3. 对比“只看照片”“只看临床资料”“照片＋临床资料”。下面还展示两种更复杂的融合模型。
4. 看数据集原有标签，再看这些模型有没有答对。标签不作为模型输入。
5. 往下看整个保留测试集的结果。一个样本答对，不代表模型整体可靠。

## AI 在哪里

这里的 AI 是训练过的分类模型，不是聊天机器人。ResNet18 先把照片变成数字特征；后面的模型学习这些图像特征、临床字段和数据集标签之间的关系。

- **只看照片：** 用图像特征做预测。
- **只看临床资料：** 用年龄、部位、症状等字段做预测。
- **简单融合：** 把两边的数字并排放进同类线性模型。这是判断“增加资料有没有帮助”的主要比较。
- **MetaBlock：** 临床资料调节图像特征的强弱，再做预测。可以理解为不同背景下，模型对照片中的某些特征给予不同权重。
- **HEALNet：** 两类信息通过注意力机制共同更新一个内部表示，再分类。这里复用了原作者的模块，缩小了模型规模，没有复现原论文的完整实验。

ResNet18 的图像编码器已经在外部 ImageNet 数据上训练过。这次冻结它，只训练后面的预测/融合部分，所以属于迁移学习的实际练习，但不是从头训练大型视觉模型。

## 怎么防止“看起来很准，其实偷看了答案”

同一个患者、可能重复使用的病变编号、完全相同的照片不能跨训练和测试分区。预处理只在训练数据上拟合。模型用内部调参集选保存哪一轮，再锁定权重后查看保留测试集。没有把测试结果拿回去继续调模型。

界面里的百分比是模型的类别得分，不是临床验证过的患病概率。真实数据不等于临床有效：这个项目还没有经过外部医院或前瞻性临床验证。

## “上线”做到了哪一步

网页实际加载模型权重并执行计算。另有本地 Python API，包含输入校验、模型版本和校验值、推理时间记录，以及一个保存审核状态的技术演示。

这证明的是研究模型的部署与服务化练习。真正医院上线还需要实际医院系统、权限、数据治理、临床验证和运营流程，本项目没有这些经历。简历可以写“independent research prototype / browser inference / versioned inference API”，不能写成在医院部署过临床诊断系统。

## 面试时可以这样说

“I built an independent research demo using paired public skin-lesion photographs and clinical metadata. I compared image-only, metadata-only and fusion models, adapted the authors' MetaBlock and HEALNet modules, and deployed the saved weights for browser inference. I kept patient groups separate and checked the exported predictions against PyTorch. This was a retrospective portfolio project, not a hospital deployment.”

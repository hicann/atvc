# 样例介绍
| 样例名                                                         | 描述                                                         |      模板           | 算子调用方式|
| ------------------------------------------------------------ | ------------------------------------------------------------- | --------------- | ------------------ |
| [add](./add/README.md) | 使用ATVC的Elementwise模板实现Add算子以及调用样例                                                    |     Elementwise             | Kernel直调 |
| [sinh_custom](./sinh_custom/README.md) | 临时Tensor参与计算的自定义Elementwise类算子以及调用样例                     |      Elementwise              | Kernel直调 |
| [add_with_scalar](./add_with_scalar/README.md) | 输入带标量的自定义Elementwise类算子以及调用样例                 |       Elementwise             | Kernel直调 |
| [reduce_sum](./reduce_sum/README.md) | 使用ATVC的Reduce模板实现自定义ReduceSum算子以及调用样例                        |       Reduce              | Kernel直调 |
| [broadcast_to](./broadcast_to/README.md) | 使用ATVC的Broadcast模板实现自定义BroadcastTo算子以及调用样例             |       Broadcast             | Kernel直调 |
| [tanh_grad](./tanh_grad/README.md) | 使用Tiling超参进行算子性能调优的ElementWise类算子调用样例                         |        Elementwise           | Kernel直调 |
| [ops_aclnn](./ops_aclnn/README.md) | 使用ATVC基于自定义工程算子的实现以及调用样例                                                    |                       | 单算子API调用 |
| [ops_pytorch](./ops_pytorch/README.md) | 使用ATVC开发自定义算子，并实现从[PyTorch](https://gitee.com/ascend/pytorch)框架调用的样例     |                      | PyTorch框架调用 |
| [add_with_broadcast](./add_with_broadcast/README.md) |使用ATVC的Elementwise和Broadcast组合模板实现Add算子以及调用样例                |       Broadcast              | Kernel直调 |
| [relu_with_reduce_sum](./relu_with_reduce_sum/README.md) |使用ATVC的Elementwise和ReduceSum组合模板实现ReduceSum+Relu算子以及调用样例|       Reduce             | Kernel直调 |
| [edge](./edge/README.md) |使用ATVC的Pool模板实现Edge算子以及调用样例|      Pool             | Kernel直调 |

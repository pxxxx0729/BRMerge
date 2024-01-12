开始
1 编译代码: 
- 确保安装成功 Intel's oneAPI base toolkit 和 HPC toolkit
- 打开 Makefile 并在 INCLUDE 变量中更正安装路径 
- 执行 $> bash make.sh 编译

2 执行 $> cmd matrix1_name [matrix2_name] [num_threads]
例如， $> ./brmerge_precise patents_main patents_main 80 计算 patents_main 矩阵与 patents_main 矩阵的乘积，使用80个 CPU 线程。


前缀 reg\_ 表示这是方法的 regression 版本，仅打印矩阵名称和 GFLOPS 结果。

前缀 mkl_ 表示这是使用了 Intel Math Kernel Library 的数学函数库，专门优化了在英特尔体系结构上运行的数学、科学和工程应用程序的性能。

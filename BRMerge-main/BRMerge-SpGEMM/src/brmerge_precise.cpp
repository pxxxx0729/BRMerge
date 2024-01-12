#include "spgemm.h"
#include "mkl_spgemm.h"

int main(int argc, char** argv) {
    // 定义存储两个矩阵名称的字符串
    std::string mat1, mat2;
    mat1 = "can_24";
    mat2 = "can_24";

    // 处理命令行参数
    if (argc == 2) {
        mat1 = argv[1];
        mat2 = argv[1];
    }
    if (argc >= 3) {
        mat1 = argv[1];
        mat2 = argv[2];
    }

    // 构建矩阵文件路径
    std::string mat1_file;
    if (mat1.find("ER") != std::string::npos) {
        mat1_file = "../matrix/ER/" + mat1 + ".mtx";
    }
    else if (mat1.find("G500") != std::string::npos) {
        mat1_file = "../matrix/G500/" + mat1 + ".mtx";
    }
    else {
        mat1_file = "../matrix/suite_sparse/" + mat1 + "/" + mat1 + ".mtx";
    }
    std::string mat2_file;
    if (mat2.find("ER") != std::string::npos) {
        mat2_file = "../matrix/ER/" + mat2 + ".mtx";
    }
    else if (mat2.find("G500") != std::string::npos) {
        mat2_file = "../matrix/G500/" + mat2 + ".mtx";
    }
    else {
        mat2_file = "../matrix/suite_sparse/" + mat2 + "/" + mat2 + ".mtx";
    }

    // 构建 CSR 对象 A 和 B
    CSR A, B;
    A.construct(mat1_file);
    if (mat1 == mat2) {
        B = A;
    }
    else {
        B.construct(mat2_file);
        if (A.N == B.M) {
            // A 和 B 的列数相等，无需调整
        }
        else if (A.N < B.M) {
            // A 的列数小于 B 的列数，构建一个新的 CSR 对象 tmp，列数为 A.N
            CSR tmp(B, A.N, B.N, 0, 0);
            B = tmp;
        }
        else {
            // A 的列数大于 B 的列数，构建一个新的 CSR 对象 tmp，列数为 B.M
            CSR tmp(A, A.M, B.M, 0, 0);
            A = tmp;
        }
    }

    // 构建 CSR 对象 C 用于存储乘法结果
    CSR C;

    // 设置线程数并计算总的乘法运算次数
    int num_threads = 64;
    if (argc >= 4) {
        num_threads = atoi(argv[3]);
    }
    omp_set_num_threads(num_threads);

    long total_flop = compute_flop(A, B);

    // 进行一次brmerge_precise运算，输出时间
    Precise_Timing timing, benchtiming;
    brmerge_precise(A, B, C, timing);
    timing.print(total_flop * 2);
    C.release();

    // 进行多次brmerge_precise运算，输出平均时间
    int iter = 10;
    for (int i = 0; i < iter; i++) {
        brmerge_precise(A, B, C, timing);
        benchtiming += timing;
        if (i < iter - 1) {
            C.release();
        }
    }
    benchtiming /= iter;
    benchtiming.print(total_flop * 2);

    CSR C_ref;
    mkl(A, B, C_ref);
    if (C == C_ref) {
        printf("pass\n");
    }
    else {
        printf("fail\n");
    }

    C.release();
}
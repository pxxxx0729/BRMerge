#ifndef _Z_RMERGE_SPGEMM_H_
#define _Z_RMERGE_SPGEMM_H_

#include "common.h"
#include "CSR.h"
#include "utils.h"

long compute_flop(mint *arpt, mint *acol, mint *brpt, mint M, mint* nnzrA, mint *floprC){
    long total_flop = 0;
#pragma omp parallel
{
    int thread_flop = 0;
#pragma omp for
    for(mint i = 0; i < M; i++){
        int local_sum = 0;
        nnzrA[i] = arpt[i+1] - arpt[i];
        for(mint j = arpt[i]; j < arpt[i+1]; j++){
            local_sum += brpt[acol[j]+1] - brpt[acol[j]];
        }
        floprC[i] = local_sum;
        thread_flop += local_sum;
    }
#pragma omp critical
{
    total_flop += thread_flop;
}
}
    return total_flop;
}

long compute_flop(mint *arpt, mint *acol, mint *brpt, mint M, mint* nnzrA, mint *floprC, mint *num_threads){
    long total_flop = 0;
#pragma omp parallel
{
    int thread_flop = 0;
#pragma omp for
    for(mint i = 0; i < M; i++){
        int local_sum = 0;
        nnzrA[i] = arpt[i+1] - arpt[i];
        for(mint j = arpt[i]; j < arpt[i+1]; j++){
            local_sum += brpt[acol[j]+1] - brpt[acol[j]];
        }
        floprC[i] = local_sum;
        thread_flop += local_sum;
    }
#pragma omp critical
{
    total_flop += thread_flop;
}
#pragma omp single
{
    *num_threads = omp_get_num_threads();
}
}
    return total_flop;
}


long compute_flop(const CSR& A, const CSR& B){
    mint *nnzrA = new mint [A.M];
    mint *row_flop = new mint [A.M];
    long flop = compute_flop(A.rpt, A.col, B.rpt, A.M, nnzrA, row_flop);
    delete [] row_flop;
    delete [] nnzrA;
    return flop;
}


inline int upper_log(int num){
    // assert(num > 0);
    int a = num;
    int base = 0;
    while(a = a>>1){
        base++;
    }
    if(likely(num != 1<<base))
        return 1 << (base+1);
    else
        return 1 << base;
}

// Prefix sum (Sequential)
template <typename T1, typename T2>
void seq_scan(T1 *in, T2 *out, int N)
{
    out[0] = 0;
    for (int i = 0; i < N - 1; ++i) {
        out[i + 1] = out[i] + in[i];
    }
}

// Prefix sum (Thread parallel)
template <typename T1, typename T2>
void scan(T1 *in, T2 *out, int N)
{
    if (N < (1 << 17)) {
        seq_scan(in, out, N);
    }
    else {
        int tnum;
        // my modify, different from yusuke, not much difference in performance
        #pragma omp parallel
        {
            #pragma omp single
            {
                tnum = omp_get_num_threads();
            }
        }
        int each_n = N / tnum;
        //T2 *partial_sum = (T *)scalable_malloc(sizeof(T) * (tnum));
        T2 *partial_sum = new T2 [tnum];
#pragma omp parallel num_threads(tnum)
        {
            int tid = omp_get_thread_num();
            int start = each_n * tid;
            int end = (tid < tnum - 1)? start + each_n : N;
            out[start] = 0;
            for (int i = start; i < end - 1; ++i) {
                out[i + 1] = out[i] + in[i];
            }
            partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier

            int offset = 0;
            for (int i = 0; i < tid; ++i) {
                offset += partial_sum[i];
            }
            for (int i = start; i < end; ++i) {
                out[i] += offset;
            }
        }
        //out[N] = out[N-1] + in[N-1];
        //scalable_free(partial_sum);
        delete [] partial_sum;
    }
}

template <typename T1, typename T2>
void seq_prefix_sum(T1 *in, T2 *out, int N)
{
    out[0] = 0;
    for (int i = 0; i < N; ++i) {
        out[i + 1] = out[i] + in[i];
    }
}

template <typename T1, typename T2>
void para_prefix_sum(T1 *in, T2 *out, int N, int tnum){

    //printf("num threads %d\n", tnum);
    int each_n = N / tnum;
    //T2 *partial_sum = (T *)scalable_malloc(sizeof(T) * (tnum));
    T2 *partial_sum = new T2 [tnum];
#pragma omp parallel num_threads(tnum)
    {
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        int end = (tid < tnum - 1)? start + each_n : N;
        out[start] = 0;
        for (int i = start; i < end - 1; ++i) {
            out[i + 1] = out[i] + in[i];
        }
        partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier
        int offset = 0;
        for (int i = 0; i < tid; ++i) {
            offset += partial_sum[i];
        }
        for (int i = start; i < end; ++i) {
            out[i] += offset;
        }
        if(tid == tnum - 1)
            out[N] = out[N-1] + in[N-1];
    }
    //scalable_free(partial_sum);
    delete [] partial_sum;
}

template <typename T1, typename T2>
inline void opt_prefix_sum(T1 *in, T2 *out, int N, int tnum = 64){
    if( N < 1 << 13){
        seq_prefix_sum(in, out, N);
    }
    else{
        para_prefix_sum(in, out, N, tnum);
    }
}

template <typename T>
int binary_search_approx(T *arr, int N, T elem){
    int s = 0;
    int e = N-1;
    assert(elem >= arr[0] && "find approx elem < smallest");
    if(elem >= arr[N-1]){
        return N-1;
    }
    while(true){
        int m = (s+e)/2;
        if(arr[m] == elem){
            return m;
        }
        else if(arr[m] < elem){
            if(elem < arr[m+1]){
                return m;
            }
            s = m;
        }
        else{ // elem < arr[m]
            if(arr[m-1] < elem){
                return m-1;
            }
            e = m;
        }
    }
}




void brmerge_upper(mint* arpt, mint* acol, mdouble* aval,
    mint* brpt, mint* bcol, mdouble* bval,
    mint** crpt_, mint** ccol_, mdouble** cval_,
    mint M, mint K, mint N, mint* cnnz_, Upper_Timing& timing) {
    double t0, t1;
    t0 = t1 = fast_clock_time();

    // 第 1 步：计算前缀和 floprC
    timing.thread = fast_clock_time() - t0;
    t0 = fast_clock_time();

    mint* nnzrA = my_malloc<mint>(M * sizeof(mint));
    mint* floprC = my_malloc<mint>(M * sizeof(mint));
    long* scan_floprC = my_malloc<long>((M + 1) * sizeof(long));
    *crpt_ = new mint[M + 1];
    mint* crpt = *crpt_;
    timing.pre_allocate = fast_clock_time() - t0;

    t0 = fast_clock_time();
    int num_threads;
    long real_total_flops = compute_flop(arpt, acol, brpt, M, nnzrA, floprC, &num_threads);
    timing.compute_flop = fast_clock_time() - t0;

    // 计算前缀和
    t0 = fast_clock_time();
    opt_prefix_sum(floprC, scan_floprC, M, num_threads);
    timing.prefix_sum_flop = fast_clock_time() - t0;

    // 第 2 步：实现负载平衡
    t0 = fast_clock_time();
    long* static_section = my_malloc<long>((num_threads + 1) * sizeof(long));
    float divide_flops = float(real_total_flops) / num_threads;
    static_section[0] = 0;
    for (int i = 1; i < num_threads; i++) {
        static_section[i] = binary_search_approx(scan_floprC, M + 1, long(divide_flops * i));
    }
    static_section[num_threads] = M;
    timing.load_balance = fast_clock_time() - t0;

    // 第 3 步：分配 C_bar和 C_bar 指针

    t0 = fast_clock_time();
    mint** col_C_bar = my_malloc<mint*>(num_threads * sizeof(mint*));
    mdouble** val_C_bar = my_malloc<mdouble*>(num_threads * sizeof(mdouble*));
    timing.allocate_hook = fast_clock_time() - t0;

    t0 = fast_clock_time();

#pragma omp parallel
    {
        int rid = omp_get_thread_num();
        int max_nnzrA = 0;
        int max_floprC = 0;
        int row_start = static_section[rid];
        int row_end = static_section[rid + 1];
        long start_rpt_C_bar = scan_floprC[row_start];
        long end_rpt_C_bar = scan_floprC[row_end];
        long local_size = end_rpt_C_bar - start_rpt_C_bar;
        for (int i = row_start; i < row_end; i++) {
            if (unlikely(max_nnzrA < nnzrA[i])) {
                max_nnzrA = nnzrA[i];
            }
            if (unlikely(max_floprC < floprC[i])) {
                max_floprC = floprC[i];
            }
        }
        // 第 4 步：分配双缓冲区，计算 C
        mint* col_ping = my_malloc<mint>(max_floprC * sizeof(mint));
        mdouble* val_ping = my_malloc<mdouble>(max_floprC * sizeof(mdouble));
        mint* list_offset_ping = my_malloc<mint>((max_nnzrA + 1) * sizeof(mint));
        mint* col_pong = my_malloc<mint>(max_floprC * sizeof(mint));
        mdouble* val_pong = my_malloc<mdouble>(max_floprC * sizeof(mdouble));
        mint* list_offset_pong = my_malloc<mint>((max_nnzrA + 1) * sizeof(mint));
        col_C_bar[rid] = my_malloc<mint>(local_size);
        val_C_bar[rid] = my_malloc<mdouble>(local_size);

        // 计算 C
        mint* dst_col, * src_col, * dst_list_offset, * src_list_offset;
        mdouble* dst_val, * src_val;

        for (int A_row = row_start; A_row < row_end; A_row++) {
            // 乘法并初始化一个双缓冲区
            dst_col = col_ping;
            dst_val = val_ping;
            dst_list_offset = list_offset_ping;
            int left_num_list = 0;
            int buffer_incr = 0;
            dst_list_offset[0] = 0;
            for (int A_idx = arpt[A_row]; A_idx < arpt[A_row + 1]; A_idx++) { // 迭代 A，n 个列表
                for (int B_idx = brpt[acol[A_idx]]; B_idx < brpt[acol[A_idx] + 1]; B_idx++) { // 一个 B 行，一个列表
                    dst_col[buffer_incr] = bcol[B_idx];
                    dst_val[buffer_incr++] = aval[A_idx] * bval[B_idx];
                }
                dst_list_offset[++left_num_list] = buffer_incr;
            }

            if (unlikely(left_num_list == 0)) {
                floprC[A_row] = 0;
                continue;
            }

            // 2路合并双缓冲区直到只剩一个
            int src1_buffer_index, src2_buffer_index, dst_buffer_index;
            src_col = col_ping;
            src_val = val_ping;
            src_list_offset = list_offset_ping;
            dst_col = col_pong;
            dst_val = val_pong;
            dst_list_offset = list_offset_pong;
            while (left_num_list != 1) {
                int inner_num_list = left_num_list;
                left_num_list = 0;
                int src_list_incr = 0;
                int dst_list_incr = 0;
                dst_list_offset[0] = 0;
                while (inner_num_list) {
                    dst_buffer_index = dst_list_offset[dst_list_incr];
                    if (inner_num_list >= 2) { // 合并两个列表
                        src1_buffer_index = src_list_offset[src_list_incr];
                        src2_buffer_index = src_list_offset[src_list_incr + 1];
                        while (src1_buffer_index < src_list_offset[src_list_incr + 1] &&
                            src2_buffer_index < src_list_offset[src_list_incr + 2]) {
                            if (unlikely(src_col[src1_buffer_index] == src_col[src2_buffer_index])) {
                                dst_col[dst_buffer_index] = src_col[src1_buffer_index];
                                dst_val[dst_buffer_index++] = src_val[src1_buffer_index++] + src_val[src2_buffer_index++];
                            }
                            else if (src_col[src1_buffer_index] < src_col[src2_buffer_index]) {
                                dst_col[dst_buffer_index] = src_col[src1_buffer_index];
                                dst_val[dst_buffer_index++] = src_val[src1_buffer_index++];
                            }
                            else {
                                dst_col[dst_buffer_index] = src_col[src2_buffer_index];
                                dst_val[dst_buffer_index++] = src_val[src2_buffer_index++];
                            }
                        }
                        int src1_num_left = src_list_offset[src_list_incr + 1] - src1_buffer_index;
                        int src2_num_left = src_list_offset[src_list_incr + 2] - src2_buffer_index;
                        if ((src1_num_left && src2_num_left)) {
                            printf("src1_num_left %d, src2_num_left %d\n", src1_num_left, src2_num_left);
                        }
                        assert(!(src1_num_left && src2_num_left) &&
                            "src1_num_left src2_num_left cant both be non zero");
                        if (src1_num_left) { // 可能还有一个列表的元素
                            memcpy(dst_col + dst_buffer_index, src_col + src1_buffer_index,
                                src1_num_left * sizeof(int));
                            memcpy(dst_val + dst_buffer_index, src_val + src1_buffer_index,
                                src1_num_left * sizeof(double));
                            dst_buffer_index += src1_num_left;
                        }
                        else if (src2_num_left) {
                            memcpy(dst_col + dst_buffer_index, src_col + src2_buffer_index,
                                src2_num_left * sizeof(int));
                            memcpy(dst_val + dst_buffer_index, src_val + src2_buffer_index,
                                src2_num_left * sizeof(double));
                            dst_buffer_index += src2_num_left;
                        }
                        src_list_incr += 2;
                        left_num_list++;
                        inner_num_list -= 2;
                        dst_list_offset[++dst_list_incr] = dst_buffer_index;
                    }
                    else if (inner_num_list == 1) { // 复制剩余的一个列表
                        src1_buffer_index = src_list_offset[src_list_incr];
                        int src1_num_left =
                            src_list_offset[src_list_incr + 1] - src_list_offset[src_list_incr];
                        memcpy(dst_col + dst_buffer_index, src_col + src1_buffer_index,
                            src1_num_left * sizeof(int));
                        memcpy(dst_val + dst_buffer_index, src_val + src1_buffer_index,
                            src1_num_left * sizeof(double));
                        dst_buffer_index += src1_num_left;
                        src_list_incr++;
                        left_num_list++;
                        inner_num_list--;
                        dst_list_offset[++dst_list_incr] = dst_buffer_index;
                    }
                } // end while(inner_num_list)

                // 交换双缓冲区
                mint* tmp_col = src_col;
                src_col = dst_col;
                dst_col = tmp_col;
                mdouble* tmp_val = src_val;
                src_val = dst_val;
                dst_val = tmp_val;
                mint* tmp_list_offset = src_list_offset;
                src_list_offset = dst_list_offset;
                dst_list_offset = tmp_list_offset;
            } // end while(left_num_list != 1)

            // 将结果复制到 C_bar 中
            floprC[A_row] = src_list_offset[1];
            memcpy(col_C_bar[rid] + scan_floprC[A_row] - start_rpt_C_bar, src_col,
                floprC[A_row] * sizeof(mint));
            memcpy(val_C_bar[rid] + scan_floprC[A_row] - start_rpt_C_bar, src_val,
                floprC[A_row] * sizeof(mdouble));
        } // end for(iter all row)

        my_free(col_ping);
        my_free(val_ping);
        my_free(list_offset_ping);
        my_free(col_pong);
        my_free(val_pong);
        my_free(list_offset_pong);
    }

    timing.compute = fast_clock_time() - t0;
    // 第 5 步：对行大小 row_size 进行前缀和
    t0 = fast_clock_time();
    opt_prefix_sum(floprC, crpt, M, num_threads);
    timing.prefix_sum_nnz = fast_clock_time() - t0;

    t0 = fast_clock_time();
    int cnnz;
    *cnnz_ = crpt[M];
    cnnz = *cnnz_;
    *cval_ = new mdouble[cnnz];
    *ccol_ = new mint[cnnz];
    mdouble* cval = *cval_;
    mint* ccol = *ccol_;
    timing.allocate_C = fast_clock_time() - t0;

    // 第 6 步：并行复制
    t0 = fast_clock_time();
    // double *copy_time = new double [num_threads];
#pragma omp parallel
    {
        int rid = omp_get_thread_num();
        int row_start = static_section[rid];
        int row_end = static_section[rid + 1];
        long start_rpt_C_bar = scan_floprC[row_start];
        long end_rpt_C_bar = scan_floprC[row_end];
        long local_size = end_rpt_C_bar - start_rpt_C_bar;

        for (int A_row = row_start; A_row < row_end; A_row++) {
            memcpy(ccol + crpt[A_row], col_C_bar[rid] + scan_floprC[A_row] - start_rpt_C_bar,
                floprC[A_row] * sizeof(mint));
            memcpy(cval + crpt[A_row], val_C_bar[rid] + scan_floprC[A_row] - start_rpt_C_bar,
                floprC[A_row] * sizeof(mdouble));
        }
        my_free(col_C_bar[rid]);
        my_free(val_C_bar[rid]);
    }
    timing.copy = fast_clock_time() - t0;

    t0 = fast_clock_time();
    my_free(col_C_bar);
    my_free(val_C_bar);
    my_free(nnzrA);
    my_free(floprC);
    my_free(scan_floprC);
    my_free(static_section);
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
}


inline void brmerge_upper(const CSR& A, const CSR& B, CSR& C, Upper_Timing &timing){
    C.M = A.M;
    C.N = B.N;
    brmerge_upper(A.rpt, A.col, A.val, B.rpt, B.col, B.val, &C.rpt, &C.col, &C.val, A.M, A.N, B.N, &C.nnz, timing);
}


void brmerge_precise(mint* arpt, mint* acol, mdouble* aval,
    mint* brpt, mint* bcol, mdouble* bval,
    mint** crpt_, mint** ccol_, mdouble** cval_,
    mint M, mint K, mint N, mint* cnnz_, Precise_Timing& timing) {

    double t0, t1;
    t0 = t1 = fast_clock_time();
    // 计算 floprC
    mint* nnzrA = my_malloc<mint>(M * sizeof(mint));
    mint* floprC = my_malloc<mint>(M * sizeof(mint));
    long* scan_floprC = my_malloc<long>((M + 1) * sizeof(long));
    *crpt_ = new mint[M + 1];
    mint* crpt = *crpt_;
    timing.pre_allocate = fast_clock_time() - t0;

    t0 = fast_clock_time();
    int num_threads;
    long real_total_flops = compute_flop(arpt, acol, brpt, M, nnzrA, floprC, &num_threads);
    timing.compute_flop = fast_clock_time() - t0;

    // 第 1 步：对 floprC 进行前缀和操作
    t0 = fast_clock_time();
    opt_prefix_sum(floprC, scan_floprC, M, num_threads);
    timing.prefix_sum_flop = fast_clock_time() - t0;

    // 第 2 步：实现负载均衡
    t0 = fast_clock_time();
    long* static_section = my_malloc<long>((num_threads + 1) * sizeof(long));
    float divide_flops = float(real_total_flops) / num_threads;
    static_section[0] = 0;
    for (int i = 1; i < num_threads; i++) {
        static_section[i] = binary_search_approx(scan_floprC, M + 1, long(divide_flops * i));
    }
    static_section[num_threads] = M;
    timing.load_balance = fast_clock_time() - t0;

    t0 = fast_clock_time();
    mint* max_floprC = my_malloc<mint>(num_threads * sizeof(mint));
    mint* max_nnzrA = my_malloc<mint>(num_threads * sizeof(mint));

    // 多个线程中并行执行
#pragma omp parallel
    {
        int rid = omp_get_thread_num();
        int row_start = static_section[rid];
        int row_end = static_section[rid + 1];
        int t_max_nnzrA = 0;
        int t_max_floprC = 0;
        // 遍历当前线程负责的行，找到最大非零元素数量和最大浮点运算数数量
        for (int i = row_start; i < row_end; i++) {
            if (unlikely(t_max_nnzrA < nnzrA[i])) {
                t_max_nnzrA = nnzrA[i];
            }
            if (unlikely(t_max_floprC < floprC[i])) {
                t_max_floprC = floprC[i];
            }
        }
        // 将结果保存到对应数组
        max_nnzrA[rid] = t_max_nnzrA;
        max_floprC[rid] = t_max_floprC;

        // 第 3 步：分配哈希表内存
        int upper_max_floprC = upper_log(t_max_floprC);
        mint* ht = my_malloc<mint>(upper_max_floprC * sizeof(mint));

        // 对每行进行计算
        for (mint A_row = row_start; A_row < row_end; A_row++) {
            // 初始化哈希表
            int ht_size = upper_log(floprC[A_row] * SYMBOLIC_SCALE);
            for (int j = 0; j < ht_size; j++) {
                ht[j] = -1;
            }
            mint nnz = 0;
            // 遍历矩阵 A 的行
            for (mint j = arpt[A_row]; j < arpt[A_row + 1]; j++) {
                mint t_acol = acol[j];
                // 遍历矩阵 B 的列
                for (mint k = brpt[t_acol]; k < brpt[t_acol + 1]; k++) {
                    mint key = bcol[k];
                    mint hash = (key * HASH_SCALE) & (ht_size - 1);
                    // 在哈希表中查找 or 插入元素
                    while (1) {
                        if (ht[hash] == key) {
                            break;
                        }
                        else if (ht[hash] == -1) {
                            ht[hash] = key;
                            nnz++;
                            break;
                        }
                        else {
                            hash = (hash + 1) & (ht_size - 1);
                        }
                    }
                }
            }
            // 通过散列方法（Hash）计算 C 矩阵的行大小，得到rpt数组和总的nnz
            floprC[A_row] = nnz;
        }
        // 释放哈希表内存
        my_free(ht);
    }
    timing.symbolic = fast_clock_time() - t0;

    // 第 4 步：对行大小进行前缀和操作并计算 C
    t0 = fast_clock_time();
    opt_prefix_sum(floprC, crpt, M, num_threads);
    timing.prefix_sum_nnz = fast_clock_time() - t0;

    t0 = fast_clock_time();
    int cnnz;
    *cnnz_ = crpt[M];
    cnnz = *cnnz_;
    *cval_ = new mdouble[cnnz];
    *ccol_ = new mint[cnnz];
    mdouble* cval = *cval_;
    mint* ccol = *ccol_;
    timing.allocate_C = fast_clock_time() - t0;

    // 计算 C
    t0 = fast_clock_time();

#pragma omp parallel
    {
        // 获取当前线程编号
        int rid = omp_get_thread_num();

        // 获取当前线程的最大 nnzrA 和 floprC
        int t_max_nnzrA = max_nnzrA[rid];
        int t_max_floprC = max_floprC[rid];

        // 获取当前线程负责的行范围
        int row_start = static_section[rid];
        int row_end = static_section[rid + 1];

        // 第 5 步：分配双缓冲区
        mint* col_ping = my_malloc<mint>(t_max_floprC * sizeof(mint));
        mdouble* val_ping = my_malloc<mdouble>(t_max_floprC * sizeof(mdouble));
        mint* list_offset_ping = my_malloc<mint>((t_max_nnzrA + 1) * sizeof(mint));
        mint* col_pong = my_malloc<mint>(t_max_floprC * sizeof(mint));
        mdouble* val_pong = my_malloc<mdouble>(t_max_floprC * sizeof(mdouble));
        mint* list_offset_pong = my_malloc<mint>((t_max_nnzrA + 1) * sizeof(mint));

        // 计算 C
        mint* dst_col, * src_col, * dst_list_offset, * src_list_offset;
        mdouble* dst_val, * src_val;

        // 遍历当前线程负责的行
        for (int A_row = row_start; A_row < row_end; A_row++) {
            // 将乘法结果存入 Ping 缓冲区
            dst_col = col_ping;
            dst_val = val_ping;
            dst_list_offset = list_offset_ping;
            int left_num_list = 0;
            int buffer_incr = 0;
            dst_list_offset[0] = 0;

            // 遍历矩阵 A 每行，生成乘法中间结果列表
            for (int A_idx = arpt[A_row]; A_idx < arpt[A_row + 1]; A_idx++) { // 遍历 A 矩阵的非零元素，形成 n 个列表
                for (int B_idx = brpt[acol[A_idx]]; B_idx < brpt[acol[A_idx] + 1]; B_idx++) { // 遍历 B 矩阵的一行，形成一个列表
                    dst_col[buffer_incr] = bcol[B_idx];
                    dst_val[buffer_incr++] = aval[A_idx] * bval[B_idx];
                }
                dst_list_offset[++left_num_list] = buffer_incr;
            }

            // 处理没有乘法结果的情况
            if (unlikely(left_num_list == 0)) {
                floprC[A_row] = 0;
                continue;
            }

            // 两两合并双缓冲区直到只剩一个列表
            int src1_buffer_index, src2_buffer_index, dst_buffer_index;
            src_col = col_ping;
            src_val = val_ping;
            src_list_offset = list_offset_ping;
            dst_col = col_pong;
            dst_val = val_pong;
            dst_list_offset = list_offset_pong;

            // 合并列表
            while (left_num_list != 1) {
                int inner_num_list = left_num_list;
                left_num_list = 0;
                int src_list_incr = 0;
                int dst_list_incr = 0;
                dst_list_offset[0] = 0;

                // 遍历内部列表，两两合并
                while (inner_num_list) {
                    dst_buffer_index = dst_list_offset[dst_list_incr];

                    // 合并两个列表
                    if (inner_num_list >= 2) {
                        src1_buffer_index = src_list_offset[src_list_incr];
                        src2_buffer_index = src_list_offset[src_list_incr + 1];

                        while (src1_buffer_index < src_list_offset[src_list_incr + 1] && src2_buffer_index < src_list_offset[src_list_incr + 2]) {
                            if (unlikely(src_col[src1_buffer_index] == src_col[src2_buffer_index])) {
                                dst_col[dst_buffer_index] = src_col[src1_buffer_index];
                                dst_val[dst_buffer_index++] = src_val[src1_buffer_index++] + src_val[src2_buffer_index++];
                            }
                            else if (src_col[src1_buffer_index] < src_col[src2_buffer_index]) {
                                dst_col[dst_buffer_index] = src_col[src1_buffer_index];
                                dst_val[dst_buffer_index++] = src_val[src1_buffer_index++];
                            }
                            else {
                                dst_col[dst_buffer_index] = src_col[src2_buffer_index];
                                dst_val[dst_buffer_index++] = src_val[src2_buffer_index++];
                            }
                        }

                        int src1_num_left = src_list_offset[src_list_incr + 1] - src1_buffer_index;
                        int src2_num_left = src_list_offset[src_list_incr + 2] - src2_buffer_index;

                        // 检查是否存在未处理的项
                        if ((src1_num_left && src2_num_left)) {
                            printf("src1_num_left %d, src2_num_left %d\n", src1_num_left, src2_num_left);
                        }

                        // 确保两个列表中只有一个非空
                        assert(!(src1_num_left && src2_num_left) && "src1_num_left src2_num_left cant both be non zero");

                        // 处理可能剩余的项
                        if (src1_num_left) {
                            memcpy(dst_col + dst_buffer_index, src_col + src1_buffer_index, src1_num_left * sizeof(int));
                            memcpy(dst_val + dst_buffer_index, src_val + src1_buffer_index, src1_num_left * sizeof(double));
                            dst_buffer_index += src1_num_left;
                        }
                        else if (src2_num_left) {
                            memcpy(dst_col + dst_buffer_index, src_col + src2_buffer_index, src2_num_left * sizeof(int));
                            memcpy(dst_val + dst_buffer_index, src_val + src2_buffer_index, src2_num_left * sizeof(double));
                            dst_buffer_index += src2_num_left;
                        }

                        src_list_incr += 2;
                        left_num_list++;
                        inner_num_list -= 2;
                        dst_list_offset[++dst_list_incr] = dst_buffer_index;
                    }
                    else if (inner_num_list == 1) { // 复制剩余的一个列表
                        src1_buffer_index = src_list_offset[src_list_incr];
                        int src1_num_left = src_list_offset[src_list_incr + 1] - src_list

                            _offset[src_list_incr];
                        memcpy(dst_col + dst_buffer_index, src_col + src1_buffer_index, src1_num_left * sizeof(int));
                        memcpy(dst_val + dst_buffer_index, src_val + src1_buffer_index, src1_num_left * sizeof(double));
                        dst_buffer_index += src1_num_left;
                        src_list_incr++;
                        left_num_list++;
                        inner_num_list--;
                        dst_list_offset[++dst_list_incr] = dst_buffer_index;
                    }
                } // end while(inner_num_list)

                // 交换双缓冲区
                mint* tmp_col = src_col;
                src_col = dst_col;
                dst_col = tmp_col;
                mdouble* tmp_val = src_val;
                src_val = dst_val;
                dst_val = tmp_val;
                mint* tmp_list_offset = src_list_offset;
                src_list_offset = dst_list_offset;
                dst_list_offset = tmp_list_offset;
            } // end while(left_num_list != 1)

            // 第 5 步：将结果复制到 C 矩阵中
            memcpy(ccol + crpt[A_row], src_col, floprC[A_row] * sizeof(mint));
            memcpy(cval + crpt[A_row], src_val, floprC[A_row] * sizeof(mdouble));
        } // end for(iter all row)

        // 释放双缓冲区
        my_free(col_ping);
        my_free(val_ping);
        my_free(list_offset_ping);
        my_free(col_pong);
        my_free(val_pong);
        my_free(list_offset_pong);
        //compute_time[rid] = fast_clock_time()  - t3;
    }

    // 计算时间
    timing.compute = fast_clock_time() - t0;

    t0 = fast_clock_time();
    my_free(max_floprC);
    my_free(max_nnzrA);
    my_free(nnzrA);
    my_free(floprC);
    my_free(scan_floprC);
    my_free(static_section);
    timing.cleanup = fast_clock_time() - t0;

    // 计算总时间
    timing.total = fast_clock_time() - t1;
}

inline void brmerge_precise(const CSR& A, const CSR& B, CSR& C, Precise_Timing &timing){
    C.M = A.M;
    C.N = B.N;
    brmerge_precise(A.rpt, A.col, A.val, B.rpt, B.col, B.val, &C.rpt, &C.col, &C.val, A.M, A.N, B.N, &C.nnz, timing);
}

#endif

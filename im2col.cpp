// 验证im2col以及col2im函数

#include<memory.h>
#include<malloc.h>
#include<stdio.h>
#include<iostream>
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) 
{
    if (alpha == 0) 
    {
        memset(Y, 0, sizeof(Dtype) * N); 
        return;
    }
    for (int i = 0; i < N; ++i)
    {
        Y[i] = alpha;
    }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col)
{
    // 计算输出的size
    const int output_h = (height + 2 * pad_h -
        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    //data_im是输入数据的指针，每遍历一次就移动channel_size的位移
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) 
    {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
        {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
            {
                //dilation_h这个变量是每隔多少个像素取值，比如dilation_h=2
                //那就是每隔2个像素取值，现在我们为了便于思考，都假设dilation_h=1
                //逐行遍历卷积窗口的输入数据
                int input_row = -pad_h + kernel_row * dilation_h;
                //逐行遍历输出数据
                for (int output_rows = output_h; output_rows; output_rows--) 
                {
                    //如果坐标超出输入数据的界限，一般出现这种情况是因为pad!=0
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) 
                    {
                        //逐列遍历输出数据，由于输入数据的行超出界限（补0)，对应的输出为0
                        for (int output_cols = output_w; output_cols; output_cols--) 
                        {
                            *(data_col++) = 0;
                        }
                    } 
                    else 
                    {
                        //逐列遍历卷积窗口的输入数据
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) 
                        {
                            //输入数据的行坐标和列坐标均没有超过界限
                            if (is_a_ge_zero_and_a_lt_b(input_col, width))
                            {
                                //那么输出的值便等于输入的值
                                *(data_col++) = data_im[input_row * width + input_col];
                            } 
                            else
                            {
                                //如果输入列坐标超过界限，便置0
                                *(data_col++) = 0;
                            }
                            //输出列坐标移动（下一个卷积窗口了）
                            input_col += stride_w;
                        }
                    }
                    //输入行坐标移动（下一个卷积窗口了）
                    input_row += stride_h;
                }
            }
        }
    }
}

void col2im_cpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im) 
{
    caffe_set(height * width * channels, float(0), data_im);
    const int output_h = (height + 2 * pad_h -
        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size)
    {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) 
        {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
            {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) 
                {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) 
                    {
                        //其他逻辑都是相同的，只是前者置0，这里就是直接跳过什么也不做
                        data_col += output_w;
                    } 
                    else
                    {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) 
                        {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width))
                            {
                                //注意这里是累加，因为这个函数一般用于卷积层的反向传播
                                //的残差传递，在卷积的前向过程中，每个输入数据是对应多个
                                //卷积窗口的，因此再反向残差传递时需要将这一对多的关系合并
                                //故进行累加
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

int main()
{
    // a 为 3*3*3的tensor
    float a[] = {1,2,0,1,1,3,0,2,2,0,2,1,0,3,2,1,1,0,1,2,1,0,1,3,3,3,2};

    float* b = (float*)malloc(sizeof(float)*48);
    // b 为12*4的tensor
    im2col_cpu(a,3,3,3,2,2,0,0,1,1,1,1,b);
    float* c = (float*)malloc(sizeof(float)*(sizeof(a)/sizeof(a[0])));
    col2im_cpu(b,3,3,3,2,2,0,0,1,1,1,1,c);


    return 0;
}
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
// ִ�������ǣ�����ִ��LayerSetUp��Ȼ��ִ��Reshape
// ������ǰ��������



// ����kernel�Ĵ�С
// ����stride
// ����pad
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  // �Ƿ���Ҫǿ��nά���
  force_nd_im2col_ = conv_param.force_nd_im2col();
  
  // ����ͼ���ͨ���ǵڼ�����
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  // һ����˵channel֮����ͼ��ĸ߶ȺͿ�ȣ������֮Ϊ�ռ���
  const int first_spatial_axis = channel_axis_ + 1;
  
  // һ���м����ᣬ����ͼ���blob��[b,c,h,w]��ônum_axes=4
  // channel_axis = 1, first_spatial_axis=2
  const int num_axes = bottom[0]->num_axes();

  // �ռ���ĸ���
  // num_spatial_axes_=4-2=2
  num_spatial_axes_ = num_axes - first_spatial_axis;

  // num_spatial_axes_ >= 0
  CHECK_GE(num_spatial_axes_, 0);

  // ������shape�Ķ��Ǵ�ŵ�����״����
  // �����blob����״��[b,c,h]����bottom_dim_blob_shape��ά����3
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  
  // �ռ��blob����״��[h,w]��������max(num_spatial_axes_,1)=2
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));

  
  // Setup filter kernel dimensions (kernel_shape_).
  // filter kernel��
  // kernel����״
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {// �����ε�kernel
    // kernel_size�е���״�����ĸ���������num_kernel_dims = 1
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
	  // kernel_shape_data[0]��[1]=conv_param.kernel_size(0)
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }

  // ���kernel_shape_data�ǲ�����ȷ���ã���kernel�ı߳�����Ϊ<0�����
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }

  
  // Setup stride dimensions (stride_).
  // ���ò���(stride)
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  // �Ƿ��Ǿ��εķ���
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {// �����εķ���
    // num_stride_dims = 1
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
	  // stride_data[0]��[1] = conv_param.stride(0)
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }

  
  // Setup pad dimensions (pad_).
  // ����pad������
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
	  // pad_data[0]��[1] = conv_param.pad[0]
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  // ������1x1�����kernel_shape_data[0]��[1]=1����stride_data[0]��[1]=1
  // ��pad_data[0]��[1]=1
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }

  
  // Configure output channels and groups.
  // ���������ͨ���Լ������Ĵ�С

  // ����ͼ���ͨ����
  channels_ = bottom[0]->shape(channel_axis_);

  // �������֮������ͼ���ͨ����
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);

  // �����Ĵ�С
  group_ = this->layer_param_.convolution_param().group();

  // ���Ϸ���
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";

  // ������ͨ�������ͨ����ֵ��
  // conv_in_channels_��conv_out_channels_
  if (reverse_dimensions()) {// �Ƿ�Ҫ��תά�ȣ��������Ϊ���룬������Ϊ���
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {// �����channel�������channel
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }


  // ����Ȩ�ص���״
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)

  // Ȩ����״ = [conv_out_channels_, conv_in_channels_/group, weight_h, weight_w]
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;// �������channel����
  for (int i = 0; i < num_spatial_axes_; ++i) { // num_spatial_axes_=2
    weight_shape.push_back(kernel_shape_data[i]);
  }

  // ƫ�õ���״ = [num_output_]
  // bias_term = 1
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
	  // �������blobs_[0]��ʾȨ�غ�[1]��ʾƫ��
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

	// ��ʼ��Ȩ��
    // Initialize and fill the weights:
    // Ȩ�ص���״����
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    // ��ʼ��ƫ��
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

  // count(1)��ʾ�ӵ�2��ά�ȵ����һ��ά��֮������ݵĸ���
  // ��kernel_dim_ = (conv_in_channels_/group)*weight_h*weight_w
  kernel_dim_ = this->blobs_[0]->count(1);

  // �����������õ�offse
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;

  
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}


// �ú�������һ�¼�������
// �ı�top����״
// ���þ�������õ�offset
// ���þ���im2col�任֮��������col_buffer_����״
// bias_multiplier_��ƫ����չ�ɾ���
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // ��һ���ռ����ǵڼ���
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  
  // num_ = batchsize
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";

  
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }

  
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  
  // ����õ�output_shape_
  // �������ͼ��ĸ߶ȺͿ��
  compute_output_shape();

  // �������top����״
  // ��batchsizeѹ�뵽top_shape(����channel֮ǰ�Ĳ��֣�Ҳ����batchsize��)
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  // �����channelѹ�뵽top_shape
  top_shape.push_back(num_output_);
  // ѹ�����ͼ��ĳ��ȺͿ��
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  // �ı�top����״
  // batchsize X conv_out_channel X out_height X out_width
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }

  // �����reverse_dimensions��=1
  if (reverse_dimensions()) {
  	// conv_out_spatial_dim_= ����ͼ��ĳ��ȺͿ��
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    // conv_out_spatial_dim_ = ���֮�����ͼ��ĳ��ȺͿ��
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }


  // ���þ�������õ�offset
  // kernel_dim_ = (conv_in_channels_/group)*weight_h*weight_w
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;

  
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  // ������������״����ά�ģ����num_spatial_axes_+1=3
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
	  // conv_input_shape_data = [����ͼ��ͨ����, ����ͼ��h, ����ͼ��w]
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      // conv_input_shape_data = [���֮�����ͼ��ͨ����, ���֮�����ͼ��h, ���֮�����ͼ��ͼ��w]
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }

  
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
	  // col_buffer_shape_ = [kernel_dim_, conv_in_spatial_dim_]
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      // col_buffer_shape_ = [kernel_dim_, conv_out_spatial_dim_]
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }

  // ����im2col�任֮��������col_buffer_����״
  col_buffer_.Reshape(col_buffer_shape_);

  // ����Ĵ�С bottom_dim_ = conv_in_channel X image_height X image_width
  bottom_dim_ = bottom[0]->count(channel_axis_);
  // ����Ĵ�С top_dim_ = conv_out_channel X out_height X out_width
  top_dim_ = top[0]->count(channel_axis_);


  // conv_in_channels_ = ����ͼ���ͨ����
  // conv_out_spatial_dim_ = ���֮�����ͼ��ĳ��ȺͿ��
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;

  
  // top_dim_ = conv_out_channel x out_height x out_width
  // bottom_dim_ = conv_in_channel x image_height x image_width
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;

  
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  // bias_multiplier_��ƫ����չ�ɾ���
  // bias_multiplier_����״��  1 x out_spatial_dim_
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}



// ���о������
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  // kernel_dim_ = input channels per-group x kernel height x kernel width
  
  if (!is_1x1_) {
    if (!skip_im2col) {
	  // ���û��1x1�����Ҳû��skip_im2col
	  // ��ʹ��conv_im2col_cpu��ʹ�þ���˻��������е�ÿһ��kernel��С��ͼ���
	  // ���һ�����������γ�һ��
      // kernrl_dim_ = input_channels*kernel height*kernel width 
	  // height = kernel_dim_
	  // width = �����ͼ��heght*�����ͼ��width
	  // �ľ���
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }

  // ʹ��caffe��cpu_gemm�����м���
  for (int g = 0; g < group_; ++g) {
  	// ����ֱ���м���
  	// conv_out_channels_ / group_��ÿ�������������channel
  	
  	// output[output_offset_ * g] = weights[weight_offset_ * g] X col_buff[col_offset_ * g]
  	// weights����״�� [conv_out_channel x kernel_dim_]
  	// col_buff����״��[kernel_dim_ x (�����ͼ��߶ȳ��Ծ����ͼ����)]
  	// output����״��Ȼ����conv_out_channel X (�����ͼ��߶ȳ��Ծ����ͼ����)
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

// ����ƫ��
// output�Ǿ���������ͼ��
// output = output + bias * bias_multiplier_
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  // output = output + bias * bias_multiplier_
  // num_output �� conv_out_channel��ֵ���
  // bias��ת��Ϊ��״:num_output_ x 1
  // bias_multiplier_����״��:1 x out_spatial_dim_
  // ��ô���������֮�����״��:num_output_ x out_spatial_dim_
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}


// ����ķ������ܼ�
// ���������������im2colת��Ϊ����Ȼ��ִ�е�ʵ���Ͼ���y=W*x
// ��y����x��ƫ���õ�W
// ��ô��������һ����ݶȾ���col_buff = W*output
// �ڶ�col_buff��col2imת��Ϊԭͼ�񼴿���ɷ���
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    // �������col_buff[col_offset_ * g] = 
    // weights[weight_offset_ * g] x output[output_offset_ * g]
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
  	// ����col2imת������
    conv_col2im_cpu(col_buff, input);
  }
}

// ����Ǽ���Ȩ�صĸ���
// ���ʵ���Ͼ���y=W*x+b
// ��ôy����W��ƫ������x
// ��ôȨ�صĸ��¾���weights += output*col_buff
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
  	// �������weights[weight_offset_ * g] += 
  	// output[output_offset_ * g] X col_buff[col_offset_ * g]
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

// ƫ�õķ���
// ��Ϊ��y=Wx+b
// y����b��ƫ����Ϊ1
// ��ô����bias += input*bias_multiplier_
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  // input����״:num_output_ * out_spatial_dim_
  // bias_multiplier_����״:out_spatial_dim_ * 1
  // bias = input*bias_multiplier_+ bias
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
